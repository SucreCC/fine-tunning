from enum import Enum


class FinetuneStageEnum(Enum):
    """微调策略枚举"""

    SFT = "sft"
    DPO = "dpo"
    RM = "rm"  # 别名
