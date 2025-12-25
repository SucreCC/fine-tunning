"""
自定义 Trainer 类
封装训练逻辑
"""
from typing import Optional, Union
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftMixedModel

from core.data.custom_dataset import get_data_collator
from core.dto.config.config_manager import ConfigManager
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomTrainer(Trainer):
    """自定义 Trainer 类，封装训练逻辑"""
    
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel, PeftMixedModel],
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset: Optional = None,
        config: Optional[ConfigManager] = None,
        training_args: Optional[TrainingArguments] = None,
    ):
        """
        初始化 CustomTrainer
        
        Args:
            model: 模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集（可选）
            config: 配置管理器（可选，如果不提供 training_args 则需要）
            training_args: 训练参数（可选，如果不提供则从 config 创建）
        """
        if training_args is None and config is None:
            raise ValueError("必须提供 training_args 或 config")
        
        if training_args is None:
            training_args = CustomTrainer._create_training_arguments(config)
        
        # 数据整理器
        data_collator = get_data_collator(tokenizer)
        
        # 初始化父类
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        self.tokenizer = tokenizer
        self.config = config
    
    @staticmethod
    def _create_training_arguments(config: ConfigManager) -> TrainingArguments:
        """
        从配置创建训练参数
        
        Args:
            config: 配置管理器
            
        Returns:
            TrainingArguments 实例
        """
        training_config = config.training_config
        model_config = config.model_config
        wandb_config = config.wandb_config
        
        # 确定报告工具
        report_to = None
        run_name = None
        if wandb_config.use_wandb:
            report_to = "wandb"
            run_name = wandb_config.wandb_run_name or wandb_config.wandb_project
        
        # 处理评估策略和保存策略
        # 如果提供了 eval_steps，则使用 steps 策略；否则使用 no 策略
        has_eval_dataset = config.dataset_config.val_path is not None and config.dataset_config.val_path.strip() != ""
        use_eval = has_eval_dataset and training_config.eval_steps is not None and training_config.eval_steps > 0
        
        if use_eval:
            evaluation_strategy = "steps"
            eval_steps = training_config.eval_steps
            save_strategy = "steps"  # 必须与 evaluation_strategy 匹配
            load_best_model_at_end = True
        else:
            evaluation_strategy = "no"
            eval_steps = None
            save_strategy = "steps"  # 保存策略仍然使用 steps
            load_best_model_at_end = False
        
        training_args = TrainingArguments(
            output_dir=model_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            lr_scheduler_type=training_config.lr_scheduler_type,
            warmup_steps=training_config.warmup_steps,
            save_steps=training_config.save_steps,
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            logging_steps=training_config.logging_steps,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            seed=training_config.seed,
            max_grad_norm=training_config.max_grad_norm,
            save_total_limit=training_config.save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            report_to=report_to,
            run_name=run_name,
        )
        
        return training_args
