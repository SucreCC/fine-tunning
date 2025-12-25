"""
微调策略枚举
根据 strategy 名称获取对应的配置类
"""
from enum import Enum
from typing import Type

from core.dto.config.finetune_config.interface.base_finetune_config import BaseFinetuneConfig
from core.dto.config.finetune_config.interface.iml.ada_lora_config import AdaLoRAConfig
from core.dto.config.finetune_config.interface.iml.ia3_config import IA3Config
from core.dto.config.finetune_config.interface.iml.p_tuning_config import PTuningConfig
from core.dto.config.finetune_config.interface.iml.prefix_tuning_config import PrefixTuningConfig


class CustomLoRAConfig:
    pass


class FinetuneStrategyEnum(Enum):
    """微调策略枚举"""

    PREFIX_TUNING = ("prefix_tuning", PrefixTuningConfig)
    P_TUNING = ("p_tuning", PTuningConfig)
    PTUNING = ("ptuning", PTuningConfig)  # 别名
    ADALORA = ("adalora", AdaLoRAConfig)
    IA3 = ("ia3", IA3Config)
    FULL = ("full", None)  # 全量微调暂时没有特殊配置
    
    def __init__(self, strategy_name: str, config_class: Type[BaseFinetuneConfig] | None):
        """
        初始化枚举值
        
        Args:
            strategy_name: 策略名称（如 "lora"）
            config_class: 对应的配置类
        """
        self.strategy_name = strategy_name
        self.config_class = config_class
    
    @classmethod
    def from_strategy_name(cls, strategy_name: str) -> "FinetuneStrategyEnum":
        """
        根据策略名称获取枚举值
        
        Args:
            strategy_name: 策略名称（如 "lora"）
            
        Returns:
            FinetuneStrategyEnum 枚举值
            
        Raises:
            ValueError: 如果策略名称不存在
        """
        for strategy in cls:
            if strategy.strategy_name == strategy_name:
                return strategy
        raise ValueError(
            f"未知的微调策略: {strategy_name}。"
            f"支持的策略: {[s.strategy_name for s in cls]}"
        )
    
    def get_config_class(self) -> Type[BaseFinetuneConfig] | None:
        """
        获取对应的配置类
        
        Returns:
            配置类，如果该策略没有特殊配置则返回 None
        """
        return self.config_class
    
    def create_config(self, config_dict: dict) -> BaseFinetuneConfig | None:
        """
        根据配置字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置对象实例，如果该策略没有特殊配置则返回 None
        """
        if self.config_class is None:
            return None
        return self.config_class.from_dict(config_dict)

