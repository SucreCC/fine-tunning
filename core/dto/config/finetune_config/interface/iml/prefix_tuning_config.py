"""
Prefix Tuning 配置类
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from core.dto.config.finetune_config.interface.base_finetuning_config import  BaseFinetuningConfig


@dataclass
class PrefixTuningConfig(BaseFinetuningConfig):
    """Prefix Tuning 配置"""
    # Prefix 长度（虚拟 token 数量）
    num_virtual_tokens: int = 20
    # Prefix 的维度（通常等于模型隐藏层维度）
    encoder_hidden_size: Optional[int] = None
    # Prefix 投影维度（如果使用两层 MLP）
    prefix_projection: bool = False
    # 任务类型（通常为 CAUSAL_LM）
    task_type: str = "CAUSAL_LM"
    
    @classmethod
    def from_dict(cls, config: dict) -> "BaseFinetuningConfig":
        """从字典创建配置对象"""
        return cls(
            type=config.get("type", "prefix_tuning"),  # 从 BaseFinetuneConfig 继承的字段
            stage=config.get("stage", "sft"),  # 从 BaseFinetuneConfig 继承的字段
            num_virtual_tokens=config.get("num_virtual_tokens", 20),
            encoder_hidden_size=config.get("encoder_hidden_size", None),
            prefix_projection=config.get("prefix_projection", False),
            task_type=config.get("task_type", "CAUSAL_LM"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_virtual_tokens": self.num_virtual_tokens,
            "encoder_hidden_size": self.encoder_hidden_size,
            "prefix_projection": self.prefix_projection,
            "task_type": self.task_type,
        }

