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

    @classmethod
    def from_yaml(cls, config_path: str) -> "ConfigManager":
        """
        从 YAML 文件加载配置
        
        Args:
            config_path: YAML 配置文件路径
            
        Returns:
            ConfigManager 实例
        """
        # 处理相对路径
        if not os.path.isabs(config_path):
            # 获取当前文件所在目录（configs/）
            current_dir = Path(__file__).parent.absolute()
            # 如果是相对路径，相对于 configs 目录
            if not config_path.startswith("configs/"):
                config_path = str(current_dir / config_path)
            else:
                # 如果已经包含 configs/，则相对于项目根目录
                project_root = current_dir.parent
                config_path = str(project_root / config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return cls.from_dict(config)

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

    def to_yaml(self, output_path: str):
        """
        保存配置到 YAML 文件
        
        Args:
            output_path: 输出文件路径
        """
        config_dict = self.to_dict()

        # 处理相对路径
        if not os.path.isabs(output_path):
            current_dir = Path(__file__).parent.absolute()
            if not output_path.startswith("configs/"):
                output_path = str(current_dir / output_path)
            else:
                project_root = current_dir.parent
                output_path = str(project_root / output_path)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

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
