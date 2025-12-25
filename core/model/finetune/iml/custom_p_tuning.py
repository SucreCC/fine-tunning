"""
P-Tuning 实现
"""
from typing import Union
from peft import (
    PromptTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftMixedModel,
)
from transformers import PreTrainedModel

from core.dto.config.finetune_config.base_finetuning_config import BaseFinetuningConfig
from core.dto.config.finetune_config.iml.p_tuning_config import PTuningConfig
from core.model.finetune.base_custom_fintuning import BaseCustomFinetuning
from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomPTuning(BaseCustomFinetuning):
    """P-Tuning 微调实现"""
    
    def setup(
        self,
        model: PreTrainedModel,
        finetune_config: BaseFinetuningConfig,
        model_config: ModelConfig,
    ) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        设置 P-Tuning
        
        Args:
            model: 基础模型
            finetune_config: 微调配置（应该是 PTuningConfig 类型）
            model_config: 模型配置
            
        Returns:
            应用了 P-Tuning 的模型
        """
        # 类型检查
        if not isinstance(finetune_config, PTuningConfig):
            raise TypeError(f"finetune_config 必须是 PTuningConfig 类型，但得到 {type(finetune_config)}")
        
        p_tuning_config: PTuningConfig = finetune_config
        
        logger.info("配置 P-Tuning...")
        
        # 如果使用量化，准备模型用于训练
        if model_config.quantization and model_config.quantization.enable:
            model = prepare_model_for_kbit_training(model)
        
        # 确定任务类型
        task_type = TaskType.CAUSAL_LM
        
        # P-Tuning 配置（P-Tuning 在 PEFT 中使用 PromptTuningConfig）
        # 注意：P-Tuning 的 encoder 相关参数在 PEFT 中可能不支持，这里使用基本配置
        peft_config = PromptTuningConfig(
            task_type=task_type,
            num_virtual_tokens=p_tuning_config.num_virtual_tokens,
        )
        
        # 应用 P-Tuning
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("P-Tuning 设置成功")
        return model
