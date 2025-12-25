"""
微调策略枚举
根据 strategy 名称获取对应的配置类 & 配置文件名
"""
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
    def get_module_path_by_type(
        cls,
        finetune_type: str,
    ) -> str:
        """
        根据 finetune_type 获取对应的模块路径
        
        Args:
            finetune_type: 微调类型名称（如 "lora", "p_tuning"）
            
        Returns:
            模块路径，格式：core.dto.config.finetune_config.interface.iml.{module_name}
            
        Raises:
            ValueError: 如果类型不存在
        """
        finetune_type = finetune_type.lower()

        for item in cls:
            if item.strategy_name == finetune_type:
                # 根据 strategy_name 生成模块名（添加 _config 后缀）
                module_name = f"{item.strategy_name}_config"
                # 返回完整的模块路径
                return f"core.dto.config.finetune_config.interface.iml.{module_name}"

        raise ValueError(
            f"Unknown finetune strategy: {finetune_type}. "
            f"Available strategies: {[i.strategy_name for i in cls]}"
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