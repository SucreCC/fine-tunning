"""
微调实现基类
所有微调策略实现都应该继承这个基类
"""
from abc import ABC, abstractmethod
from typing import Union
from transformers import PreTrainedModel
from peft import PeftModel, PeftMixedModel
from core.dto.config.finetune_config.base_finetuning_config import BaseFinetuningConfig
from core.dto.config.model_config import ModelConfig


class BaseCustomFinetuning(ABC):
    """微调实现基类"""
    
    @abstractmethod
    def setup(
        self,
        model: PreTrainedModel,
        finetune_config: BaseFinetuningConfig,
        model_config: ModelConfig,
    ) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        设置微调策略
        
        Args:
            model: 基础模型
            finetune_config: 微调配置
            model_config: 模型配置
            
        Returns:
            应用了微调策略的模型
        """
        pass

