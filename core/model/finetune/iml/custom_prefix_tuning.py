"""
Prefix Tuning 实现
"""
from typing import Union
from peft import (
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftMixedModel,
)
from transformers import PreTrainedModel

from core.dto.config.finetune_config.base_finetuning_config import BaseFinetuningConfig
from core.model.finetune.base_custom_fintuning import BaseCustomFinetuning
from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomPrefixTuning(BaseCustomFinetuning):
    """Prefix Tuning 微调实现"""
    
    def setup(
        self,
        model: PreTrainedModel,
        finetune_config: BaseFinetuningConfig,
        model_config: ModelConfig,
    ) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        设置 Prefix Tuning
        
        Args:
            model: 基础模型
            finetune_config: 微调配置（应该是 PrefixTuningConfig 类型）
            model_config: 模型配置
            
        Returns:
            应用了 Prefix Tuning 的模型
        """
        # 类型检查
        if not isinstance(finetune_config, PrefixTuningConfig):
            raise TypeError(f"finetune_config 必须是 PrefixTuningConfig 类型，但得到 {type(finetune_config)}")
        
        prefix_tuning_config: PrefixTuningConfig = finetune_config
        
        logger.info("配置 Prefix Tuning...")
        
        # 如果使用量化，准备模型用于训练
        if model_config.quantization and model_config.quantization.enable:
            model = prepare_model_for_kbit_training(model)
        
        # 确定任务类型
        task_type = TaskType.CAUSAL_LM
        
        # Prefix Tuning 配置
        peft_config = PrefixTuningConfig(
            task_type=task_type,
            num_virtual_tokens=prefix_tuning_config.num_virtual_tokens,
            encoder_hidden_size=prefix_tuning_config.encoder_hidden_size,
            prefix_projection=prefix_tuning_config.prefix_projection,
        )
        
        # 应用 Prefix Tuning
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("Prefix Tuning 设置成功")
        return model
