"""
配置工具模块
统一配置读取
"""
import os
import sys
from pathlib import Path
from typing import Optional

# 添加 configs 目录到路径
project_root = Path(__file__).parent.parent.parent
configs_dir = project_root / "configs"
sys.path.insert(0, str(configs_dir))

from config_manager import ConfigManager

__all__ = ["ConfigManager", "load_config"]


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径
        
    Returns:
        ConfigManager 实例
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = str(configs_dir / "config.yaml")
    
    # 处理相对路径
    if not os.path.isabs(config_path):
        config_path = str(project_root / config_path)
    
    return ConfigManager.from_yaml(config_path)

