"""
微调策略枚举
根据 strategy 名称获取对应的配置类 & 配置文件名
"""
import importlib
import types
from enum import Enum
from typing import Type, Optional

from core.dto.config.finetune_config.interface.base_finetune_config import BaseFinetuneConfig
from core.dto.config.finetune_config.interface.iml.ia3_config import IA3Config
from core.dto.config.finetune_config.interface.iml.lora_config import LoRAConfig
from core.dto.config.finetune_config.interface.iml.p_tuning_config import PTuningConfig
from core.dto.config.finetune_config.interface.iml.prefix_tuning_config import PrefixTuningConfig


class FinetuneStrategyEnum(Enum):
    """微调策略枚举"""

    PREFIX_TUNING = ("prefix_tuning", PrefixTuningConfig, "ia3_config")
    P_TUNING = ("p_tuning", PTuningConfig, "p_tuning.yaml")
    PTUNING = ("ptuning", PTuningConfig, "p_tuning.yaml")   # 别名，共用配置文件
    LORA = ("lora", LoRAConfig, "lora.yaml")
    IA3 = ("ia3", IA3Config, "ia3.yaml")
    FULL = ("full", None, None)  # 全量微调无独立配置文件

    def __init__(
        self,
        strategy_name: str,
        config_class: Optional[Type[BaseFinetuneConfig]],
        module_name: str,
    ):
        self.strategy_name = strategy_name
        self.config_class = config_class
        self.module_name = module_name

    # ---------- 你原有的方法，保持不变 ----------

    @classmethod
    def get_finetune_class_by_type(
        cls,
        finetune_type: str,
    ) -> types.ModuleType:
        """
        根据 finetune_type 动态导入并返回对应的配置模块
        
        Args:
            finetune_type: 微调类型名称（如 "lora", "p_tuning", "prefix_tuning", "ia3"）
            
        Returns:
            导入的模块对象，可以通过模块获取配置类（如 module.LoRAConfig）
            
        Raises:
            ValueError: 如果类型不存在或不支持
            ModuleNotFoundError: 如果模块导入失败
            AttributeError: 如果模块中不存在对应的配置类
            
        Example:
            >>> module = FinetuneStrategyEnum.get_module_path_by_type("lora")
            >>> config_class = getattr(module, "LoRAConfig")
            >>> config = config_class.from_dict({"r": 8, "lora_alpha": 32})
        """
        finetune_type = finetune_type.lower()

        # 检查是否是 FULL 策略（全量微调，没有独立配置模块）
        if finetune_type == "full":
            raise ValueError(
                f"策略 '{finetune_type}' 是全量微调，没有独立的配置模块。"
                f"请使用其他策略: {[i.strategy_name for i in cls if i.strategy_name != 'full']}"
            )

        # 查找对应的枚举项
        for item in cls:
            if item.strategy_name == finetune_type:
                # 如果该策略没有配置类，抛出错误
                if item.config_class is None:
                    raise ValueError(
                        f"策略 '{finetune_type}' 没有对应的配置模块。"
                    )
                
                # 根据 strategy_name 生成模块名（添加 _config 后缀）
                module_name = f"{item.strategy_name}_config"
                # 构建完整的模块路径
                module_path = f"core.dto.config.finetune_config.interface.iml.{module_name}"
                
                try:
                    # 动态导入模块
                    module = importlib.import_module(module_path)
                    return module
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        f"无法找到配置模块: {module_path}。"
                        f"请确保模块文件存在于 core.dto.config.finetune_config.interface.iml 目录中"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"导入模块 {module_path} 时发生错误: {str(e)}"
                    ) from e

        # 如果没有找到对应的策略
        available_strategies = [i.strategy_name for i in cls if i.config_class is not None]
        raise ValueError(
            f"未知的微调策略: {finetune_type}。"
            f"支持的策略: {available_strategies}"
        )

    @classmethod
    def from_type(cls, finetune_type: str) -> "FinetuneStrategyEnum":
        finetune_type = finetune_type.lower()

        for item in cls:
            if item.strategy_name == finetune_type:
                return item

        raise ValueError(
            f"Unknown finetune strategy: {finetune_type}. "
            f"Available strategies: {[i.strategy_name for i in cls]}"
        )