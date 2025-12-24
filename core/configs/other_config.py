"""
其他配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OtherConfig:
    """其他配置"""
    # 是否使用 DeepSpeed（需要安装 deepspeed）
    use_deepspeed: bool = False
    # DeepSpeed 配置文件路径
    deepspeed_config: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "OtherConfig":
        """从字典创建配置对象"""
        return cls(
            use_deepspeed=config.get("use_deepspeed", False),
            deepspeed_config=config.get("deepspeed_config"),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "use_deepspeed": self.use_deepspeed,
            "deepspeed_config": self.deepspeed_config,
        }

