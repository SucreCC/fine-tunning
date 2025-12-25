"""
数据模块
"""
from .custom_dataset import CustomDataset, get_data_collator
from .interface.base_process import BaseProcess
from .interface.iml.default_process import DefaultProcess
from .interface.iml.chatglm_process import ChatGLMProcess
from .interface.iml.qwen_process import QwenProcess

__all__ = [
    "CustomDataset",
    "get_data_collator",
    "BaseProcess",
    "DefaultProcess",
    "ChatGLMProcess",
    "QwenProcess",
]

