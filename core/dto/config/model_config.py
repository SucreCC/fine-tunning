"""
模型配置类
"""
from dataclasses import dataclass
from typing import Optional
from .quantization_config import QuantizationConfig


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型路径（可以是 HuggingFace 模型名称或本地路径）
    base_model_path: str = ""
    # 输出模型保存路径
    output_dir: str = ""
    # 量化配置
    quantization: Optional[QuantizationConfig] = None

    def __post_init__(self):
        """初始化默认值"""
        if self.quantization is None:
            self.quantization = QuantizationConfig()

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        """从字典创建配置对象"""
        quantization_config = config.get("quantization", {})
        return cls(
            base_model_path=config.get("base_model_path", ""),
            output_dir=config.get("output_dir", ""),
            quantization=QuantizationConfig.from_dict(quantization_config),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "base_model_path": self.base_model_path,
            "output_dir": self.output_dir,
            "quantization": self.quantization.to_dict() if self.quantization else QuantizationConfig().to_dict(),
        }

