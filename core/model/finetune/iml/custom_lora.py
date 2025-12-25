"""
LoRA / QLoRA 实现
"""
from typing import Union
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftMixedModel,
)
from transformers import PreTrainedModel

from core.model.finetune.base_custom_fintuning import BaseCustomFinetuning
from core.dto.config.finetune_config.interface.base_finetuning_config import BaseFinetuningConfig
from core.dto.config.finetune_config.interface.iml.lora_config import LoRAConfig
from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomLoRA(BaseCustomFinetuning):
    """LoRA 微调实现"""
    
    def setup(
        self,
        model: PreTrainedModel,
        finetune_config: BaseFinetuningConfig,
        model_config: ModelConfig,
    ) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        设置 LoRA
        
        Args:
            model: 基础模型
            finetune_config: 微调配置（应该是 LoRAConfig 类型）
            model_config: 模型配置
            
        Returns:
            应用了 LoRA 的模型
        """
        # 类型检查
        if not isinstance(finetune_config, LoRAConfig):
            raise TypeError(f"finetune_config 必须是 LoRAConfig 类型，但得到 {type(finetune_config)}")
        
        lora_config: LoRAConfig = finetune_config
        
        # 如果未启用，直接返回模型
        if not lora_config.enable:
            logger.info("LoRA 未启用，使用全量模型训练")
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
            bias=lora_config.bias,
        )
        
        # 应用 LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("LoRA 设置成功")
        return model
