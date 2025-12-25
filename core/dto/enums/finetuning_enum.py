from enum import Enum


class FinetuneEnum(Enum):
    """微调策略枚举"""

    PREFIX_TUNING = ("prefix_tuning", CustomePrefixTuning, "ia3_config")
    P_TUNING = ("p_tuning", CustomePTuning, "p_tuning.yaml")
    PTUNING = ("ptuning", CustomePTuning, "p_tuning.yaml")   # 别名，共用配置文件
    LORA = ("lora", CustomeLoRA, "lora.yaml")
    IA3 = ("ia3", ICustomeA3, "ia3.yaml")
    FULL = ("full", None, None)  # 全量微调无独立配置文件