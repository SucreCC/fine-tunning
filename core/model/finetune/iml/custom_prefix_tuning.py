custom_lora.py"""
LoRA / QLoRA 注入模块
"""
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType, PeftModel, PeftMixedModel
)
from transformers import PreTrainedModel
from core.config.custom_lora_config import CustomLoRAConfig
from core.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)

def setup_lora(model: PreTrainedModel, lora_config: CustomLoRAConfig, model_config: ModelConfig) -> PreTrainedModel | PeftModel | PeftMixedModel:
    """
    设置 LoRA
    
    Args:
        model: 基础模型
        config: 配置管理器
        
    Returns:
        应用了 LoRA 的模型
    """

    
    if not lora_config.use_lora:
        return model
    
    logger.info("配置 LoRA...")
    
    # 如果使用量化，准备模型用于训练
    if model_config.quantization and model_config.quantization.enable:
        model = prepare_model_for_kbit_training(model)
    
    # 确定任务类型
    task_type = TaskType.CAUSAL_LM
    
    # LoRA 配置
    peft_config = LoraConfig(
        task_type=task_type,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias
    )
    
    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

