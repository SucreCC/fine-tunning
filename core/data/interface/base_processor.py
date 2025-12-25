"""
基础处理接口
所有数据处理类都应该继承这个基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseProcessor(ABC):
    """数据处理基类接口"""
    
    @staticmethod
    def preprocess_text(text: str, max_length: Optional[int] = None) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            max_length: 最大长度（可选）
            
        Returns:
            预处理后的文本
        """
        # 去除首尾空白
        text = text.strip()
        
        # 可以在这里添加其他预处理逻辑
        # 例如：去除特殊字符、标准化等
        
        return text
    
    @abstractmethod
    def process(self, item: Dict) -> str:
        """
        处理单个数据项，将其转换为模型输入格式
        
        Args:
            item: 包含 system 和 conversation 的字典
            
        Returns:
            格式化后的对话字符串
        """
        pass

