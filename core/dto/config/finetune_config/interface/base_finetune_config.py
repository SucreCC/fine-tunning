"""
微调配置基类
所有微调策略配置都应该继承这个基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.dto.enums.finetune_config_enum import FinetuneStrategyEnum


@dataclass
class BaseFinetuneConfig(ABC):
    """微调配置基类"""
    type: str
    stage: str = "sft"

    @classmethod
    def from_dict(cls, finetune_config: dict) -> "BaseFinetuneConfig":
        """
        从字典创建配置对象

        Args:
            finetune_config: 配置字典，包含 type、stage 和对应的策略配置

        Returns:
            配置对象实例
        """
        # 延迟导入以避免循环导入
        from core.dto.enums.finetune_config_enum import FinetuneStrategyEnum
        
        finetune_type = finetune_config.get("type")
        if not finetune_type:
            raise ValueError("finetune 配置中缺少 'type' 字段")
        
        stage = finetune_config.get("stage", "sft")
        
        # 获取对应的配置模块
        finetune_module = FinetuneStrategyEnum.get_finetune_class_by_type(finetune_type)
        
        # 从模块中获取配置类（类名格式：LoRAConfig, PTuningConfig 等）
        # 根据 finetune_type 生成类名：lora -> LoRAConfig, p_tuning -> PTuningConfig
        class_name = _get_config_class_name(finetune_type)
        
        if not hasattr(finetune_module, class_name):
            raise ValueError(
                f"模块 {finetune_module.__name__} 中找不到配置类 {class_name}。"
                f"请确保模块中存在对应的配置类"
            )
        
        config_class = getattr(finetune_module, class_name)
        
        # 获取该策略的配置字典（例如 finetune_config.get("lora")）
        strategy_config_dict = finetune_config.get(finetune_type, {})
        
        # 将 type 和 stage 添加到配置字典中，因为子类继承自 BaseFinetuneConfig 需要这些字段
        strategy_config_dict = strategy_config_dict.copy() if strategy_config_dict else {}
        strategy_config_dict["type"] = finetune_type
        strategy_config_dict["stage"] = stage
        
        # 创建配置对象
        config_instance = config_class.from_dict(strategy_config_dict)
        
        return config_instance


def _get_config_class_name(finetune_type: str) -> str:
    """
    根据 finetune_type 生成配置类名
    
    Args:
        finetune_type: 微调类型（如 "lora", "p_tuning"）
        
    Returns:
        配置类名（如 "LoRAConfig", "PTuningConfig"）
    """
    # 特殊映射
    type_to_class_name = {
        "lora": "LoRAConfig",
        "qlora": "LoRAConfig",  # QLoRA 也使用 LoRAConfig
        "p_tuning": "PTuningConfig",
        "ptuning": "PTuningConfig",  # 别名
        "prefix_tuning": "PrefixTuningConfig",
        "ia3": "IA3Config",
        "adalora": "AdaLoRAConfig",
    }
    
    # 如果存在特殊映射，直接返回
    if finetune_type in type_to_class_name:
        return type_to_class_name[finetune_type]
    
    # 否则将下划线分隔的名称转换为 PascalCase
    parts = finetune_type.split("_")
    # 每个部分首字母大写
    pascal_case = "".join(word.capitalize() for word in parts)
    # 添加 Config 后缀
    return f"{pascal_case}Config"

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        pass
