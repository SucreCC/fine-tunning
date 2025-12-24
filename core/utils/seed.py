"""
随机种子设置工具
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

