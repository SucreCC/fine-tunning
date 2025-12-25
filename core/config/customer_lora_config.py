"""
LoRA 配置类
"""
from dataclasses import dataclass
from typing import List, Optional, Literal, cast


@dataclass
class CustomerLoRAConfig:
    """LoRA 配置"""
    # 是否使用 LoRA
    use_lora: bool = True
    # LoRA rank
    r: int = 8
    # LoRA alpha
    lora_alpha: int = 32
    # LoRA target modules（根据模型类型调整）
    target_modules: Optional[List[str]] = None
    # LoRA dropout
    lora_dropout: float = 0.1
    # LoRA bias 类型（关键修改点）
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
    def from_dict(cls, config: dict) -> "CustomerLoRAConfig":
        target_modules = config.get(
            "target_modules",
            ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        )

        bias = cast(
            Literal["none", "all", "lora_only"],
            config.get("bias", "none"),
        )

        return cls(
            use_lora=config.get("use_lora", True),
            r=config.get("r", 8),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=target_modules if isinstance(target_modules, list) else list(target_modules),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias=bias,
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "use_lora": self.use_lora,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
        }
