"""
IA3 实现
"""
from typing import Union
from peft import (
    IA3Config,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftMixedModel,
)
from transformers import PreTrainedModel

from core.model.finetune.base_custom_fintuning import BaseCustomFinetuning
from core.dto.config.finetune_config.interface.base_finetuning_config import BaseFinetuningConfig
from core.dto.config.finetune_config.interface.iml.ia3_config import IA3Config as IA3ConfigDTO
from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomIA3(BaseCustomFinetuning):
    """IA3 微调实现"""
    
    def setup(
        self,
        model: PreTrainedModel,
        finetune_config: BaseFinetuningConfig,
        model_config: ModelConfig,
    ) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        设置 IA3
        
        Args:
            model: 基础模型
            finetune_config: 微调配置（应该是 IA3Config 类型）
            model_config: 模型配置
            
        Returns:
            应用了 IA3 的模型
        """
        # 类型检查
        if not isinstance(finetune_config, IA3ConfigDTO):
            raise TypeError(f"finetune_config 必须是 IA3Config 类型，但得到 {type(finetune_config)}")
        
        ia3_config: IA3ConfigDTO = finetune_config
        
        # 如果未启用，直接返回模型
        if not ia3_config.enable:
            logger.info("IA3 未启用，使用全量模型训练")
            return model
        
        logger.info("配置 IA3...")
        
        # 如果使用量化，准备模型用于训练
        if model_config.quantization and model_config.quantization.enable:
            model = prepare_model_for_kbit_training(model)
        
        # 确定任务类型
        task_type = TaskType.CAUSAL_LM
        
        # IA3 配置
        peft_config = IA3Config(
            task_type=task_type,
            target_modules=ia3_config.target_modules,
            feedforward_modules=ia3_config.feedforward_modules,
        )
        
        # 应用 IA3
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("IA3 设置成功")
        return model
