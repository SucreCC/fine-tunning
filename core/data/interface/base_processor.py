"""
基础处理接口
所有数据处理类都应该继承这个基类
"""
import importlib
from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.dto.config.dataset_config import DatasetConfig
from core.utils import logging
logger = logging.get_logger(__name__)

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
    
    @staticmethod
    def _module_name_to_class_name(module_name: str) -> str:
        """
        将模块名（snake_case）转换为类名（PascalCase）
        
        Args:
            module_name: 模块名，如 "chatglm_processor"
            
        Returns:
            类名，如 "ChatglmProcessor"
        """
        # 去掉 "_processor" 后缀
        if module_name.endswith("_processor"):
            base_name = module_name[:-10]  # 去掉 "_processor"
        else:
            base_name = module_name
        
        # 通用转换：snake_case -> PascalCase
        # 例如: chatglm -> Chatglm
        # 例如: default -> Default
        # 例如: qwen -> Qwen
        parts = base_name.split('_')
        class_name = ''.join(word.capitalize() for word in parts) + 'Processor'
        
        return class_name
    
    @staticmethod
    def get_processor(dataset_config: DatasetConfig) -> "BaseProcessor":
        """
        根据 dataset_config 中的 processor 模块名动态获取对应的 processor 实例
        
        使用模块名格式（带下划线），如：
        - "chatglm_processor"
        - "default_processor"
        - "qwen_processor"
        
        Args:
            dataset_config: 数据集配置，包含 processor 模块名和 system_prompt
            
        Returns:
            BaseProcessor 实例
            
        Raises:
            ValueError: 如果 processor 模块名不存在或无效
            ImportError: 如果无法导入对应的模块
        """
        module_name = dataset_config.processor or "default_processor"
        
        # 确保以 "_processor" 结尾
        if not module_name.endswith("_processor"):
            module_name = f"{module_name}_processor"
        
        # 构建模块路径
        module_path = f"core.data.interface.iml.{module_name}"
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            # 从模块名转换为类名
            class_name = BaseProcessor._module_name_to_class_name(module_name)
            
            # 获取类对象
            if not hasattr(module, class_name):
                raise ValueError(
                    f"模块 {module_path} 中找不到 processor 类。"
                    f"期望的类名: {class_name}"
                )
            
            processor_class = getattr(module, class_name)

            logger.info(f"使用 processor: {class_name}")
            
            # 验证是否是 BaseProcessor 的子类
            if not issubclass(processor_class, BaseProcessor):
                raise ValueError(
                    f"类 {class_name} 不是 BaseProcessor 的子类"
                )
            
            # 创建 processor 实例
            return processor_class(system_template=dataset_config.system_prompt)
            
        except ModuleNotFoundError as e:
            raise ValueError(
                f"无法找到 processor 模块: {module_path}。"
                f"请确保 processor 名称 '{module_name}' 正确，"
                f"且对应的模块文件存在于 core.data.interface.iml 目录中"
            ) from e
        except AttributeError as e:
            raise ValueError(
                f"模块 {module_path} 中不存在类。"
                f"请确保模块中存在对应的 Processor 类"
            ) from e
    
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

