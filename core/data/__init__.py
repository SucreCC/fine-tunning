"""
数据模块
"""
from .customer_dataset import CustomerDataset, get_data_collator
from .base_process import BaseProcess
from .processes import DefaultProcess, ChatGLMProcess, QwenProcess

__all__ = [
    "CustomerDataset",
    "get_data_collator",
    "BaseProcess",
    "DefaultProcess",
    "ChatGLMProcess",
    "QwenProcess",
]

