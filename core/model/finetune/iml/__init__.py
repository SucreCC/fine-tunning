"""
微调实现类模块
"""
from .custom_lora import CustomLoRA
from .custom_p_tuning import CustomPTuning
from .custom_prefix_tuning import CustomPrefixTuning
from .custom_ia3 import CustomIA3

__all__ = [
    "CustomLoRA",
    "CustomPTuning",
    "CustomPrefixTuning",
    "CustomIA3",
]

