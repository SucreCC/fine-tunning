"""
量化配置类
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class QuantizationConfig:
    """量化配置"""
    # 是否启用量化
    enable: bool = False
    # 量化位数（4 或 8）
    bits: int = 4
    # 计算数据类型
    compute_dtype: Literal["fp16", "bf16"] = "bf16"

    @classmethod
    def from_dict(cls, config: dict) -> "QuantizationConfig":
        """从字典创建配置对象"""
        return cls(
            enable=config.get("enable", False),
            bits=config.get("bits", 4),
            compute_dtype=config.get("compute_dtype", "bf16"),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "enable": self.enable,
            "bits": self.bits,
            "compute_dtype": self.compute_dtype,
        }

