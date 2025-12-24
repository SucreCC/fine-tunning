"""
训练模块
"""
from .trainer import create_trainer
from .callbacks import get_callbacks

__all__ = ["create_trainer", "get_callbacks"]

