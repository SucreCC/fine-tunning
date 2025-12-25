"""
ChatGLM 格式处理类
"""
from typing import Dict

from core.data.interface.base_processor import BaseProcessor


class ChatGLMProcessor(BaseProcessor):
    """ChatGLM 格式处理类"""
    
    def __init__(self, system_template: str = ""):
        """
        初始化 ChatGLM 处理类
        
        Args:
            system_template: 系统提示模板
        """
        self.system_template = self.preprocess_text(system_template)
    
    def process(self, item: Dict) -> str:
        """
        处理单个数据项，使用 ChatGLM 格式
        
        Args:
            item: 包含 system 和 conversation 的字典
            
        Returns:
            格式化后的对话字符串
        """
        system = item.get("system", self.system_template)
        system = self.preprocess_text(system)
        conversations = item.get("conversation", [])
        
        # ChatGLM 格式：[Round 1]\n\n问：...\n\n答：...
        formatted_text = f"[Round 0]\n\n问：系统提示：{system}\n\n答：好的，我明白了。\n\n"
        
        round_num = 1
        for conv in conversations:
            human = self.preprocess_text(conv.get("human", ""))
            assistant = self.preprocess_text(conv.get("assistant", ""))
            
            formatted_text += f"[Round {round_num}]\n\n问：{human}\n\n答：{assistant}\n\n"
            round_num += 1
        
        return formatted_text.strip()

