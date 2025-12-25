"""
微调配置基类
所有微调策略配置都应该继承这个基类
"""
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.dto.enums.finetune_config_enum import FinetuneStrategyEnum


@dataclass
class BaseFinetuneConfig(ABC):
    """微调配置基类"""
    type: "FinetuneStrategyEnum"
    stage: str = "sft"

    @classmethod
    def from_dict(cls, finetune_config: dict) -> "BaseFinetuneConfig":
        """
        从字典创建配置对象

        Args:
            config: 配置字典

        Returns:
            配置对象实例
        """

        type = finetune_config.get("type")
        stage = finetune_config.get("stage")
        finetune_config_module = FinetuneStrategyEnum.get_finetune_class_by_type(type)
        finetune_config = finetune_config_module.from_dict(finetune_config.get(type))
        finetune_config.type = type
        finetune_config.stage = stage
        return finetune_config


    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        pass
