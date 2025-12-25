"""
Trainer 封装模块
"""
from transformers import Trainer, TrainingArguments
from typing import Optional
from core.config.config_manager import ConfigManager
from core.data.dataset import get_data_collator


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset: Optional = None,
    config: Optional[ConfigManager] = None,
    training_args: Optional[TrainingArguments] = None
) -> Trainer:
    """
    创建 Trainer
    
    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 验证数据集（可选）
        config: 配置管理器
        training_args: 训练参数（可选，如果不提供则从 config 创建）
        
    Returns:
        Trainer 实例
    """
    if training_args is None and config is None:
        raise ValueError("必须提供 training_args 或 config")
    
    if training_args is None:
        training_args = create_training_arguments(config)
    
    # 数据整理器
    data_collator = get_data_collator(tokenizer)
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    return trainer


def create_training_arguments(config: ConfigManager) -> TrainingArguments:
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
        eval_steps=training_config.eval_steps if training_config.eval_steps else None,
        logging_steps=training_config.logging_steps,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        seed=training_config.seed,
        max_grad_norm=training_config.max_grad_norm,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=True if training_config.eval_steps else False,
        report_to=report_to,
        run_name=run_name,
    )
    
    return training_args

