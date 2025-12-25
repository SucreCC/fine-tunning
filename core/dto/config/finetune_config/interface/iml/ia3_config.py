"""
IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) 配置类
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from core.dto.config.finetune_config.interface.base_finetune_config import BaseFinetuneConfig


@dataclass
class IA3Config(BaseFinetuneConfig):
    """IA3 配置"""
    # 是否启用 IA3
    enable: bool = True
    # 目标模块（通常只包含 attention 和 feed-forward 层）
    target_modules: Optional[List[str]] = None
    # feed-forward 模块（用于指定哪些模块应用 IA3）
    feedforward_modules: Optional[List[str]] = None
    # 任务类型
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        """初始化默认值"""
        if self.target_modules is None:
            self.target_modules = ["k_proj", "v_proj", "out_proj"]
        if self.feedforward_modules is None:
            self.feedforward_modules = []
    
    @classmethod
    def from_dict(cls, config: dict) -> "IA3Config":
        """从字典创建配置对象"""
        target_modules = config.get(
            "target_modules",
            ["k_proj", "v_proj", "out_proj"],
        )
        feedforward_modules = config.get("feedforward_modules", [])
        
        return cls(
            enable=config.get("enable", True),
            target_modules=target_modules if isinstance(target_modules, list) else list(target_modules),
            feedforward_modules=feedforward_modules if isinstance(feedforward_modules, list) else list(feedforward_modules),
            task_type=config.get("task_type", "CAUSAL_LM"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enable": self.enable,
            "target_modules": self.target_modules,
            "feedforward_modules": self.feedforward_modules,
            "task_type": self.task_type,
        }

