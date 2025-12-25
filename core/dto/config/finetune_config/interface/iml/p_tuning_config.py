"""
P-Tuning 配置类
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal

from core.dto.config.finetune_config.interface.base_finetuning_config import BaseFinetuningConfig


@dataclass
class PTuningConfig(BaseFinetuningConfig):
    """P-Tuning 配置"""
    # 虚拟 token 数量
    num_virtual_tokens: int = 20
    # 编码器隐藏层维度
    encoder_hidden_size: Optional[int] = None
    # 编码器层数
    encoder_num_layers: int = 2
    # 编码器重排列（是否使用重排列层）
    encoder_reparameterization_type: Literal["MLP", "LSTM"] = "MLP"
    # 任务类型
    task_type: str = "CAUSAL_LM"
    
    @classmethod
    def from_dict(cls, config: dict) -> "BaseFinetuningConfig":
        """从字典创建配置对象"""
        return cls(
            type=config.get("type", "p_tuning"),  # 从 BaseFinetuneConfig 继承的字段
            stage=config.get("stage", "sft"),  # 从 BaseFinetuneConfig 继承的字段
            num_virtual_tokens=config.get("num_virtual_tokens", 20),
            encoder_hidden_size=config.get("encoder_hidden_size", None),
            encoder_num_layers=config.get("encoder_num_layers", 2),
            encoder_reparameterization_type=config.get("encoder_reparameterization_type", "MLP"),
            task_type=config.get("task_type", "CAUSAL_LM"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_virtual_tokens": self.num_virtual_tokens,
            "encoder_hidden_size": self.encoder_hidden_size,
            "encoder_num_layers": self.encoder_num_layers,
            "encoder_reparameterization_type": self.encoder_reparameterization_type,
            "task_type": self.task_type,
        }

