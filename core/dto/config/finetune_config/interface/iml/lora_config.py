"""
AdaLoRA 配置类
AdaLoRA 是 LoRA 的改进版本，可以自适应调整 rank
"""
from dataclasses import dataclass
from typing import List, Optional, Literal, cast, Dict, Any

from core.dto.config.finetune_config.interface.base_finetuning_config import BaseFinetuningConfig


@dataclass
class LoRAConfig(BaseFinetuningConfig):
    """AdaLoRA 配置"""
    # 初始 rank
    r: int = 8
    # LoRA alpha
    lora_alpha: int = 32
    # 目标模块
    target_modules: Optional[List[str]] = None
    # LoRA dropout
    lora_dropout: float = 0.1
    # 初始化范围（用于权重初始化）
    init_r: int = 12
    # 目标 rank（最终要达到的 rank）
    target_r: int = 8
    # beta1（用于优化器）
    beta1: float = 0.85
    # beta2（用于优化器）
    beta2: float = 0.85
    # 任务类型
    task_type: str = "CAUSAL_LM"
    # bias 类型
    bias: Literal["none", "all", "lora_only"] = "none"
    
    def __post_init__(self):
        """初始化默认值"""
        if self.target_modules is None:
            self.target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
    
    @classmethod
    def from_dict(cls, config: dict) -> "BaseFinetuningConfig":
        """从字典创建配置对象"""
        target_modules = config.get(
            "target_modules",
            ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        )
        
        bias = cast(
            Literal["none", "all", "lora_only"],
            config.get("bias", "none"),
        )
        
        return cls(
            type=config.get("type", "lora"),  # 从 BaseFinetuneConfig 继承的字段
            stage=config.get("stage", "sft"),  # 从 BaseFinetuneConfig 继承的字段
            r=config.get("r", 8),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=target_modules if isinstance(target_modules, list) else list(target_modules),
            lora_dropout=config.get("lora_dropout", 0.1),
            init_r=config.get("init_r", 12),
            target_r=config.get("target_r", 8),
            beta1=config.get("beta1", 0.85),
            beta2=config.get("beta2", 0.85),
            task_type=config.get("task_type", "CAUSAL_LM"),
            bias=bias,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "init_r": self.init_r,
            "target_r": self.target_r,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "task_type": self.task_type,
            "bias": self.bias,
        }

