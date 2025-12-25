"""
训练模块
"""
from .custom_trainer import CustomTrainer
from .callbacks import get_callbacks

__all__ = ["CustomTrainer", "get_callbacks"]

