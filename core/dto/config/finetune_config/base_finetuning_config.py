"""
微调配置基类
所有微调策略配置都应该继承这个基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.dto.enums.finetuning_config_enum import FinetuneStrategyEnum


@dataclass
class BaseFinetuningConfig(ABC):
    """微调配置基类"""
    type: str
    stage: str = "sft"
    enable: bool = True

    @classmethod
    def from_dict(cls, finetune_config: dict) -> "BaseFinetuningConfig":
        """
        从字典创建配置对象

        Args:
            finetune_config: 配置字典，包含 type、stage 和对应的策略配置

        Returns:
            配置对象实例
        """
        # 延迟导入以避免循环导入
        from core.dto.enums.finetuning_config_enum import FinetuneStrategyEnum
        
        finetune_type = finetune_config.get("type")
        if not finetune_type:
            raise ValueError("finetune 配置中缺少 'type' 字段")
        
        stage = finetune_config.get("stage", "sft")
        
        # 获取对应的配置类
        config_class = FinetuneStrategyEnum.get_finetune_class_by_type(finetune_type)
        
        # 获取该策略的配置字典（例如 finetune_config.get("lora")）
        strategy_config_dict = finetune_config.get(finetune_type, {})
        
        # 将 type 和 stage 添加到配置字典中，因为子类继承自 BaseFinetuneConfig 需要这些字段
        strategy_config_dict = strategy_config_dict.copy() if strategy_config_dict else {}
        strategy_config_dict["type"] = finetune_type
        strategy_config_dict["stage"] = stage
        
        # 创建配置对象
        config_instance = config_class.from_dict(strategy_config_dict)
        
        return config_instance

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        pass
