"""
模型配置类
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from .quantization_config import QuantizationConfig


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型路径（可以是 HuggingFace 模型名称或本地路径）
    base_model_path: str = ""
    # 检查点基础目录（所有检查点的根目录）
    checkpoint_dir: Optional[str] = None
    # 运行名称/实验名称（会在 checkpoint_dir 下创建以此命名的子目录）
    run_name: Optional[str] = None
    # 输出模型保存路径（完整路径，如果设置了 checkpoint_dir 和 run_name，则会被忽略）
    # 保留此字段以向后兼容，如果设置了 checkpoint_dir 和 run_name，会自动组合
    output_dir: Optional[str] = None
    # 量化配置
    quantization: Optional[QuantizationConfig] = None

    def __post_init__(self):
        """初始化默认值"""
        if self.quantization is None:
            self.quantization = QuantizationConfig()

    def get_output_dir(self) -> str:
        """
        获取完整的输出目录路径
        
        优先级：
        1. 如果设置了 checkpoint_dir 和 run_name，则组合它们
        2. 如果只设置了 output_dir，则使用 output_dir
        3. 如果只设置了 checkpoint_dir，则使用 checkpoint_dir
        4. 否则返回空字符串
        """
        if self.checkpoint_dir and self.run_name:
            # 组合 checkpoint_dir 和 run_name
            return str(Path(self.checkpoint_dir) / self.run_name)
        elif self.output_dir:
            # 向后兼容：使用 output_dir
            return self.output_dir
        elif self.checkpoint_dir:
            # 只有 checkpoint_dir，没有 run_name
            return self.checkpoint_dir
        else:
            return ""

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        """从字典创建配置对象"""
        quantization_config = config.get("quantization", {})
        return cls(
            base_model_path=config.get("base_model_path", ""),
            checkpoint_dir=config.get("checkpoint_dir"),
            run_name=config.get("run_name"),
            output_dir=config.get("output_dir"),  # 向后兼容
            quantization=QuantizationConfig.from_dict(quantization_config),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {
            "base_model_path": self.base_model_path,
            "quantization": self.quantization.to_dict() if self.quantization else QuantizationConfig().to_dict(),
        }
        # 如果设置了新字段，优先使用新字段
        if self.checkpoint_dir is not None:
            result["checkpoint_dir"] = self.checkpoint_dir
        if self.run_name is not None:
            result["run_name"] = self.run_name
        # 向后兼容：如果只设置了 output_dir，也保存它
        if self.output_dir is not None and self.checkpoint_dir is None:
            result["output_dir"] = self.output_dir
        return result

