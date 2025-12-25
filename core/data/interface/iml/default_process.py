"""
默认处理类
"""
from typing import Dict

from core.data.interface.base_process import BaseProcess


class DefaultProcess(BaseProcess):
    """默认处理类"""
    
    def __init__(self, system_template: str = ""):
        """
        初始化默认处理类
        
        Args:
            system_template: 系统提示模板
        """
        self.system_template = self.preprocess_text(system_template)
    
    def process(self, item: Dict) -> str:
        """
        处理单个数据项，使用默认格式
        
        Args:
            item: 包含 system 和 conversation 的字典
            
        Returns:
            格式化后的对话字符串
        """
        system = item.get("system", self.system_template)
        system = self.preprocess_text(system)
        conversations = item.get("conversation", [])
        
        # 默认通用格式
        formatted_text = f"<system>\n{system}\n</system>\n\n"
        
        for conv in conversations:
            human = self.preprocess_text(conv.get("human", ""))
            assistant = self.preprocess_text(conv.get("assistant", ""))
            
            formatted_text += f"<human>\n{human}\n</human>\n\n"
            formatted_text += f"<assistant>\n{assistant}\n</assistant>\n\n"
        
        return formatted_text.strip()

