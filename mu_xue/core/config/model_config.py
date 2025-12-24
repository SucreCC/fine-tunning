"""
模型配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型路径（可以是 HuggingFace 模型名称或本地路径）
    base_model_path: str = "THUDM/chatglm3-6b"
    # 输出模型保存路径
    output_dir: str = "../model"
    # 是否使用 8bit 量化（节省显存）
    use_8bit: bool = False
    # 是否使用 4bit 量化（更节省显存）
    use_4bit: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        """从字典创建配置对象"""
        return cls(
            base_model_path=config.get("base_model_path", "THUDM/chatglm3-6b"),
            output_dir=config.get("output_dir", "../model"),
            use_8bit=config.get("use_8bit", False),
            use_4bit=config.get("use_4bit", False),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "base_model_path": self.base_model_path,
            "output_dir": self.output_dir,
            "use_8bit": self.use_8bit,
            "use_4bit": self.use_4bit,
        }

