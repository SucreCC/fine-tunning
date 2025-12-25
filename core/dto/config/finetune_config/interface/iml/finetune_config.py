"""
微调策略配置类
"""
from dataclasses import dataclass
from typing import Literal, Optional, Any, Dict, TYPE_CHECKING
from core.dto.config.finetune_config.interface.base_finetune_config import BaseFinetuneConfig


@dataclass
class FinetuneConfig:
    """微调策略配置"""
    # 微调策略：full（全量微调）、lora、qlora、prefix_tuning、p_tuning、adalora、ia3
    strategy: Literal["full", "lora", "qlora", "prefix_tuning", "p_tuning", "ptuning", "adalora", "ia3"] = "lora"
    # 微调阶段：sft（监督微调）、dpo（直接偏好优化）、rm（奖励模型）
    stage: Literal["sft", "dpo", "rm"] = "sft"
    # 策略特定配置（根据 strategy 动态加载）
    strategy_config: Optional[BaseFinetuneConfig] = None

    @classmethod
    def from_dict(cls, config: dict, FinetuneStrategyEnum=None) -> "FinetuneConfig":
        """
        从字典创建配置对象
        
        根据 strategy 自动加载对应的策略配置
        """
        strategy = config.get("strategy", "lora")
        stage = config.get("stage", "sft")

        # 根据 strategy 获取对应的枚举和配置类
        try:
            strategy_enum = FinetuneStrategyEnum.from_strategy_name(strategy)
        except ValueError:
            # 如果策略不存在，使用默认的 lora
            strategy_enum = FinetuneStrategyEnum.LORA
            strategy = "lora"
        
        # 获取策略配置字典（从 finetune 下的对应键获取）
        strategy_config_dict = {}
        if strategy_enum.config_class is not None:
            # 根据策略名称获取配置字典
            strategy_key_map = {
                "lora": "lora",
                "qlora": "lora",  # QLoRA 也使用 lora 配置
                "prefix_tuning": "prefix_tuning",
                "p_tuning": "p_tuning",
                "ptuning": "p_tuning",  # 别名
                "adalora": "adalora",
                "ia3": "ia3",
            }
            config_key = strategy_key_map.get(strategy, strategy)
            strategy_config_dict = config.get(config_key, {})
            
            # 创建策略配置对象
            strategy_config = strategy_enum.create_config(strategy_config_dict)
        else:
            strategy_config = None
        
        return cls(
            strategy=strategy,
            stage=stage,
            strategy_config=strategy_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "strategy": self.strategy,
            "stage": self.stage,
        }
        
        # 添加策略特定配置
        if self.strategy_config is not None:
            strategy_key_map = {
                "lora": "lora",
                "qlora": "lora",
                "prefix_tuning": "prefix_tuning",
                "p_tuning": "p_tuning",
                "ptuning": "p_tuning",
                "adalora": "adalora",
                "ia3": "ia3",
            }
            config_key = strategy_key_map.get(self.strategy, self.strategy)
            result[config_key] = self.strategy_config.to_dict()
        
        return result

