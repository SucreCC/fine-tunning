"""
配置管理器
统一管理所有配置类
"""
import os
import yaml
from pathlib import Path
from typing import Optional

from .model_config import ModelConfig
from .dataset_config import DatasetConfig
from .training_config import TrainingConfig
from .lora_config import LoRAConfig
from .other_config import OtherConfig


class ConfigManager:
    """配置管理器类"""
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        other_config: Optional[OtherConfig] = None,
    ):
        """初始化配置管理器"""
        self.model_config = model_config or ModelConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.training_config = training_config or TrainingConfig()
        self.lora_config = lora_config or LoRAConfig()
        self.other_config = other_config or OtherConfig()

    @classmethod
    def from_dict(cls, config: dict) -> "ConfigManager":
        """从字典创建配置管理器"""
        return cls(
            model_config=ModelConfig.from_dict(config.get("model", {})),
            dataset_config=DatasetConfig.from_dict(config.get("dataset", {})),
            training_config=TrainingConfig.from_dict(config.get("training", {})),
            lora_config=LoRAConfig.from_dict(config.get("lora", {})),
            other_config=OtherConfig.from_dict(config.get("other", {})),
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
            # 获取当前文件所在目录（core/config/）
            current_dir = Path(__file__).parent.absolute()
            # 获取 mu_xue 目录
            mu_xue_dir = current_dir.parent.parent
            config_path = str(mu_xue_dir / config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls.from_dict(config)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model": self.model_config.to_dict(),
            "dataset": self.dataset_config.to_dict(),
            "training": self.training_config.to_dict(),
            "lora": self.lora_config.to_dict(),
            "other": self.other_config.to_dict(),
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
            mu_xue_dir = current_dir.parent.parent
            output_path = str(mu_xue_dir / output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return (
            f"ConfigManager(\n"
            f"  model_config={self.model_config},\n"
            f"  dataset_config={self.dataset_config},\n"
            f"  training_config={self.training_config},\n"
            f"  lora_config={self.lora_config},\n"
            f"  other_config={self.other_config}\n"
            f")"
        )

