"""
微调策略配置类
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class FinetuneConfig:
    """微调策略配置"""
    # 微调策略：full（全量微调）、lora、qlora
    strategy: Literal["full", "lora", "qlora"] = "lora"
    # 微调阶段：sft（监督微调）、dpo（直接偏好优化）、rm（奖励模型）
    stage: Literal["sft", "dpo", "rm"] = "sft"

    @classmethod
    def from_dict(cls, config: dict) -> "FinetuneConfig":
        """从字典创建配置对象"""
        return cls(
            strategy=config.get("strategy", "lora"),
            stage=config.get("stage", "sft"),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "strategy": self.strategy,
            "stage": self.stage,
        }

