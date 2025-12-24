"""
数据集配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 训练集路径
    train_path: str = ""
    # 验证集路径
    val_path: Optional[str] = ""
    # 最大序列长度
    max_length: int = 2048
    # 是否使用流式加载（节省内存）
    streaming: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> "DatasetConfig":
        """从字典创建配置对象"""
        return cls(
            train_path=config.get("train_path", ""),
            val_path=config.get("val_path", ""),
            max_length=config.get("max_length"),
            streaming=config.get("streaming", False),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "train_path": self.train_path,
            "val_path": self.val_path,
            "max_length": self.max_length,
            "streaming": self.streaming,
        }

