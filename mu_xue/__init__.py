"""
沐雪模型训练模块
"""
from .train import train
from .dataset import ConversationDataset

__version__ = "1.0.0"
__all__ = ["train", "ConversationDataset"]

