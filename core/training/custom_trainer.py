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

# 延迟导入回调类，避免循环导入
try:
    import wandb
except ImportError:
    wandb = None


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
            training_args = CustomTrainer._create_training_arguments(config, model)
        
        # 数据整理器
        data_collator = get_data_collator(tokenizer)
        
        # 检查模型是否已经在多个设备上（通过 device_map="auto"）
        self._model_on_multiple_devices = False
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            self._model_on_multiple_devices = True
        elif hasattr(model, 'device_map') and model.device_map:
            self._model_on_multiple_devices = True
        
        # 初始化父类
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 使用 processing_class 代替已弃用的 tokenizer
        # 新版本的 transformers 推荐使用 processing_class
        # 直接设置 processing_class，如果 Trainer 不支持会自动回退到 tokenizer
        try:
            self.processing_class = tokenizer
        except AttributeError:
            # 向后兼容：如果 processing_class 不存在，使用 tokenizer
            self.tokenizer = tokenizer
        
        self.config = config
        
        # 自动添加回调（如果提供了 config）
        if config is not None:
            self._add_callbacks(config)
    
    def _add_callbacks(self, config: ConfigManager):
        """
        根据配置自动添加回调
        
        Args:
            config: 配置管理器
        """
        # 添加 Wandb 回调
        if config.wandb_config.use_wandb:
            from core.training.callbacks import WandbCallback
            self.add_callback(WandbCallback(config.wandb_config))
            logger.info("已添加 Wandb 回调")
    
    def _setup_devices(self):
        """
        重写设备设置方法，如果模型已经在多个设备上，跳过设备移动以避免警告
        """
        if self._model_on_multiple_devices:
            # 模型已经在多个设备上，不需要移动
            # 直接使用模型的当前设备配置
            import torch
            if torch.cuda.is_available():
                self.args._n_gpu = torch.cuda.device_count()
            else:
                self.args._n_gpu = 0
            return
        else:
            # 正常设置设备
            super()._setup_devices()
    
    @staticmethod
    def _create_training_arguments(config: ConfigManager, model=None) -> TrainingArguments:
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
            eval_strategy = "steps"
            eval_steps = training_config.eval_steps
            save_strategy = "steps"  # 必须与 eval_strategy 匹配
            load_best_model_at_end = True
        else:
            eval_strategy = "no"
            eval_steps = None
            save_strategy = "steps"  # 保存策略仍然使用 steps
            load_best_model_at_end = False
        
        # 检查模型是否已经在多个设备上（通过 device_map="auto"）
        # 如果是，设置 dataloader_pin_memory=False 以避免警告
        model_on_multiple_devices = False
        if model is not None:
            # 检查模型是否有 hf_device_map 属性（表示使用了 device_map="auto"）
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                model_on_multiple_devices = True
            # 或者检查模型是否有 device_map 属性
            elif hasattr(model, 'device_map') and model.device_map:
                model_on_multiple_devices = True
        
        # 处理设备配置
        device_config = training_config.device
        import os
        import torch
        
        # 检查是否使用 CUDA
        use_cuda = device_config.use_cuda if device_config.use_cuda is not None else torch.cuda.is_available()
        
        # 处理并行策略配置
        parallel_strategy = device_config.parallel_strategy
        ddp_backend = device_config.ddp_backend
        local_rank = None
        
        # 如果设置了并行策略，自动设置相应的后端
        if parallel_strategy == "ddp":
            # DDP 策略：自动选择后端
            if ddp_backend is None:
                ddp_backend = "nccl" if use_cuda else "gloo"
        elif parallel_strategy == "deepspeed":
            # DeepSpeed 策略：需要 deepspeed 配置文件
            if device_config.deepspeed_config is None:
                logger.warning("使用 DeepSpeed 策略但未指定 deepspeed_config，将尝试自动查找")
        elif parallel_strategy == "fsdp":
            # FSDP 策略：使用 PyTorch FSDP
            logger.info("使用 FSDP (Fully Sharded Data Parallel) 策略")
        
        # 处理分布式训练配置
        if (ddp_backend or parallel_strategy == "ddp") and use_cuda:
            # 从环境变量获取 local_rank（如果存在）
            local_rank = device_config.local_rank
            if local_rank is None:
                local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        # 构建 TrainingArguments 参数字典
        training_args_dict = {
            "output_dir": model_config.output_dir,
            "num_train_epochs": training_config.num_epochs,
            "per_device_train_batch_size": training_config.per_device_train_batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "lr_scheduler_type": training_config.lr_scheduler_type,
            "warmup_steps": training_config.warmup_steps,
            "save_steps": training_config.save_steps,
            "save_strategy": save_strategy,
            "eval_steps": eval_steps,
            "logging_steps": training_config.logging_steps,
            "fp16": training_config.fp16,
            "bf16": training_config.bf16,
            "seed": training_config.seed,
            "max_grad_norm": training_config.max_grad_norm,
            "save_total_limit": training_config.save_total_limit,
            "load_best_model_at_end": load_best_model_at_end,
            "report_to": report_to,
            "run_name": run_name,
        }
        
        # 添加设备相关配置
        # 使用 use_cpu 代替已弃用的 no_cuda
        if not use_cuda:
            training_args_dict["use_cpu"] = True
        
        # 添加并行策略配置
        if parallel_strategy == "ddp":
            # DDP 配置
            if ddp_backend:
                training_args_dict["ddp_backend"] = ddp_backend
            training_args_dict["ddp_find_unused_parameters"] = device_config.ddp_find_unused_parameters
            training_args_dict["ddp_timeout"] = device_config.ddp_timeout
            if local_rank is not None and local_rank >= 0:
                training_args_dict["local_rank"] = local_rank
        elif parallel_strategy == "deepspeed":
            # DeepSpeed 配置
            if device_config.deepspeed_config:
                training_args_dict["deepspeed"] = device_config.deepspeed_config
            else:
                # 尝试从环境变量或默认路径查找
                deepspeed_config_path = os.environ.get("DEEPSPEED_CONFIG_FILE", "deepspeed_config.json")
                if os.path.exists(deepspeed_config_path):
                    training_args_dict["deepspeed"] = deepspeed_config_path
                    logger.info(f"使用 DeepSpeed 配置文件: {deepspeed_config_path}")
                else:
                    logger.warning("未找到 DeepSpeed 配置文件，将使用默认配置")
        elif parallel_strategy == "fsdp":
            # FSDP 配置
            if device_config.fsdp_config:
                training_args_dict["fsdp"] = device_config.fsdp_config
            else:
                # 默认 FSDP 配置
                training_args_dict["fsdp"] = {
                    "fsdp_transformer_layer_cls_to_wrap": None,  # 自动检测
                    "fsdp_backward_prefetch": "BACKWARD_PRE",
                    "fsdp_forward_prefetch": False,
                    "fsdp_use_orig_params": True,
                }
                logger.info("使用默认 FSDP 配置")
        
        # 如果模型已经在多个设备上，设置 dataloader_pin_memory=False
        # 这样可以避免 Trainer 尝试移动模型到单个设备
        if model_on_multiple_devices:
            training_args_dict["dataloader_pin_memory"] = False
        
        # 根据 transformers 版本使用正确的参数名
        # 旧版本使用 eval_strategy，新版本使用 evaluation_strategy
        try:
            # 先尝试使用 eval_strategy（旧版本 transformers）
            training_args = TrainingArguments(eval_strategy=eval_strategy, **training_args_dict)
        except TypeError:
            # 如果失败，尝试使用 evaluation_strategy（新版本 transformers）
            training_args = TrainingArguments(evaluation_strategy=eval_strategy, **training_args_dict)
        
        return training_args
