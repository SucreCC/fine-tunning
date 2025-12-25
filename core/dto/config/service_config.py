"""
其他配置类
"""
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """其他配置"""
    service_name: str = ""
    description: str = ""
    version: str = ""

    @classmethod
    def from_dict(cls, config: dict) -> "ServiceConfig":
        """从字典创建配置对象"""
        return cls(
            service_name=config.get("service_name", ""),
            description=config.get("description", ""),
            version=config.get("version", ""),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "service_name": self.service_name,
            "description": self.description,
            "version": self.version,
        }
