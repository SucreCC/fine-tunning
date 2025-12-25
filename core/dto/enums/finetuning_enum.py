"""
微调实现枚举
根据 strategy 名称获取对应的微调实现类
"""
import importlib
from enum import Enum
from typing import Type, Optional, TYPE_CHECKING

from core.model.finetune.base_custom_fintuning import BaseCustomFinetuning

if TYPE_CHECKING:
    from core.model.finetune.iml.custom_lora import CustomLoRA
    from core.model.finetune.iml.custom_p_tuning import CustomPTuning
    from core.model.finetune.iml.custom_prefix_tuning import CustomPrefixTuning
    from core.model.finetune.iml.custom_ia3 import CustomIA3


class FinetuneEnum(Enum):
    """微调实现枚举"""

    PREFIX_TUNING = ("prefix_tuning", "CustomPrefixTuning", "custom_prefix_tuning")
    P_TUNING = ("p_tuning", "CustomPTuning", "custom_p_tuning")
    PTUNING = ("ptuning", "CustomPTuning", "custom_p_tuning")  # 别名，共用实现
    LORA = ("lora", "CustomLoRA", "custom_lora")
    IA3 = ("ia3", "CustomIA3", "custom_ia3")
    FULL = ("full", None, None)  # 全量微调无独立实现

    def __init__(
        self,
        strategy_name: str,
        class_name: Optional[str],
        module_name: Optional[str],
    ):
        self.strategy_name = strategy_name
        self.class_name = class_name  # 类名字符串，用于动态导入
        self.module_name = module_name

    @classmethod
    def get_finetune_class_by_type(
        cls,
        finetune_type: str,
    ) -> Type[BaseCustomFinetuning]:
        """
        根据 finetune_type 动态导入并返回对应的微调实现类
        
        Args:
            finetune_type: 微调类型名称（如 "lora", "p_tuning", "prefix_tuning", "ia3"）
            
        Returns:
            微调实现类（如 CustomLoRA, CustomPTuning）
            
        Raises:
            ValueError: 如果类型不存在或不支持
            ModuleNotFoundError: 如果模块导入失败
            AttributeError: 如果模块中不存在对应的实现类
            
        Example:
            >>> finetune_class = FinetuneEnum.get_finetune_class_by_type("lora")
            >>> finetune_instance = finetune_class()
            >>> model = finetune_instance.setup(model, config, model_config)
        """
        finetune_type = finetune_type.lower()

        # 检查是否是 FULL 策略（全量微调，没有独立实现）
        if finetune_type == "full":
            raise ValueError(
                f"策略 '{finetune_type}' 是全量微调，没有独立的实现类。"
                f"请使用其他策略: {[i.strategy_name for i in cls if i.strategy_name != 'full']}"
            )

        # 查找对应的枚举项
        for item in cls:
            if item.strategy_name == finetune_type:
                # 如果该策略没有实现类，抛出错误
                if item.module_name is None or item.class_name is None:
                    raise ValueError(
                        f"策略 '{finetune_type}' 没有对应的实现模块。"
                    )
                
                # 构建完整的模块路径
                module_path = f"core.model.finetune.iml.{item.module_name}"
                
                try:
                    # 动态导入模块
                    module = importlib.import_module(module_path)
                    
                    # 检查模块中是否存在实现类
                    if not hasattr(module, item.class_name):
                        raise AttributeError(
                            f"模块 {module_path} 中找不到实现类 {item.class_name}。"
                            f"请确保模块中存在对应的实现类"
                        )
                    
                    # 获取实现类
                    finetune_class = getattr(module, item.class_name)
                    return finetune_class
                    
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        f"无法找到实现模块: {module_path}。"
                        f"请确保模块文件存在于 core.model.finetune.iml 目录中"
                    ) from e
                except AttributeError:
                    raise  # 重新抛出 AttributeError
                except Exception as e:
                    raise RuntimeError(
                        f"导入模块 {module_path} 时发生错误: {str(e)}"
                    ) from e

        # 如果没有找到对应的策略
        available_strategies = [i.strategy_name for i in cls if i.module_name is not None]
        raise ValueError(
            f"未知的微调策略: {finetune_type}。"
            f"支持的策略: {available_strategies}"
        )

    @classmethod
    def from_type(cls, finetune_type: str) -> "FinetuneEnum":
        """
        根据 finetune_type 获取对应的枚举值
        
        Args:
            finetune_type: 微调类型名称
            
        Returns:
            FinetuneEnum 枚举值
        """
        finetune_type = finetune_type.lower()

        for item in cls:
            if item.strategy_name == finetune_type:
                return item

        raise ValueError(
            f"Unknown finetune strategy: {finetune_type}. "
            f"Available strategies: {[i.strategy_name for i in cls]}"
        )
