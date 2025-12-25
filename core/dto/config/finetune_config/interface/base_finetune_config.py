"""
微调配置基类
所有微调策略配置都应该继承这个基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

from core.dto.enums import FinetuneStrategyEnum


@dataclass
class BaseFinetuneConfig(ABC):
    """微调配置基类"""
    type: FinetuneStrategyEnum
    stage: str = "finetune"
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict) -> "BaseFinetuneConfig":
        """
        从字典创建配置对象
        
        Args:
            config: 配置字典
            
        Returns:
            配置对象实例
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        pass

