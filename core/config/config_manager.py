"""
配置管理器
统一管理所有配置类
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .dataset_config import DatasetConfig
from .log_config import LogConfig
from .lora_config import LoRAConfig
from .model_config import ModelConfig
from .service_config import ServiceConfig
from .training_config import TrainingConfig
from .wandb_config import WandbConfig


@dataclass
class ConfigManager:
    """配置管理器类"""
    service_config: Optional[ServiceConfig] = None,
    log_config: Optional[LogConfig] = None,
    model_config: Optional[ModelConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    lora_config: Optional[LoRAConfig] = None,
    wandb_config: Optional[WandbConfig] = None,

    @staticmethod
    def find_project_root(config_file_name: str = "config.yaml") -> Path:
        """
        查找项目根路径（config.yaml 所在的文件夹）

        从当前工作目录一级一级往上找，直到找到指定的配置文件

        Args:
            config_file_name: 配置文件名，默认为 "config.yaml"

        Returns:
            项目根目录的绝对路径

        Raises:
            FileNotFoundError: 如果找不到配置文件
        """
        # 从当前工作目录开始向上查找
        current_path = Path(os.getcwd()).absolute()

        # 向上查找，直到找到包含配置文件的目录
        while current_path != current_path.parent:
            config_file = current_path / config_file_name
            if config_file.exists():
                return current_path
            current_path = current_path.parent

        # 如果找不到，抛出异常
        raise FileNotFoundError(
            f"找不到配置文件 '{config_file_name}'。请确保在项目根目录或其子目录中运行。"
        )

    @classmethod
    def from_dict(cls, config: dict) -> "ConfigManager":
        """从字典创建配置管理器"""
        return cls(
            service_config=ServiceConfig.from_dict(config.get("service", {})),
            log_config=LogConfig.from_dict(config.get("log", {})),
            model_config=ModelConfig.from_dict(config.get("model", {})),
            dataset_config=DatasetConfig.from_dict(config.get("dataset", {})),
            training_config=TrainingConfig.from_dict(config.get("training", {})),
            lora_config=LoRAConfig.from_dict(config.get("lora", {})),
            wandb_config=WandbConfig.from_dict(config.get("wandb", {})),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "service": self.service_config.to_dict(),
            "log": self.log_config.to_dict(),
            "model": self.model_config.to_dict(),
            "dataset": self.dataset_config.to_dict(),
            "training": self.training_config.to_dict(),
            "lora": self.lora_config.to_dict(),
            "wandb": self.wandb_config.to_dict(),
        }

    @classmethod
    def from_yaml(cls, config_file_name: str = "config.yaml") -> "ConfigManager":
        """
        从 YAML 文件加载配置
        
        调用 find_project_root 找到项目根目录，然后拼接配置文件的绝对路径并读取
        
        Args:
            config_file_name: 配置文件名，默认为 "config.yaml"
            
        Returns:
            ConfigManager 实例
        """
        # 调用 find_project_root 找到项目根目录
        project_root = cls.find_project_root(config_file_name)
        
        # 拼接配置文件的绝对路径
        config_path = str(project_root / config_file_name)
        
        # 读取 yaml 文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 返回 ConfigManager 实例
        return cls.from_dict(config)

    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return (
            f"ConfigManager(\n"
            f"  service_config={self.service_config},\n"
            f"  log_config={self.log_config},\n"
            f"  model_config={self.model_config},\n"
            f"  dataset_config={self.dataset_config},\n"
            f"  training_config={self.training_config},\n"
            f"  lora_config={self.lora_config},\n"
            f"  wandb_config={self.wandb_config}\n"
            f")"
        )
