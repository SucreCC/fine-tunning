"""
Qwen 格式处理类
"""
from typing import Dict

from core.data.interface.base_processor import BaseProcessor


class QwenProcessor(BaseProcessor):
    """Qwen 格式处理类"""
    
    def __init__(self, system_template: str = ""):
        """
        初始化 Qwen 处理类
        
        Args:
            system_template: 系统提示模板
        """
        self.system_template = self.preprocess_text(system_template)
    
    def process(self, item: Dict) -> str:
        """
        处理单个数据项，使用 Qwen 格式
        
        Args:
            item: 包含 system 和 conversation 的字典
            
        Returns:
            格式化后的对话字符串
        """
        system = item.get("system", self.system_template)
        system = self.preprocess_text(system)
        conversations = item.get("conversation", [])
        
        # Qwen 格式使用特殊 token
        formatted_text = f"<|im_start|>system\n{system}<|im_end|>\n"
        
        for conv in conversations:
            human = self.preprocess_text(conv.get("human", ""))
            assistant = self.preprocess_text(conv.get("assistant", ""))
            
            formatted_text += f"<|im_start|>user\n{human}<|im_end|>\n"
            formatted_text += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        
        return formatted_text

