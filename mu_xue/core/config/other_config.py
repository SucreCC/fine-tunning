"""
其他配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OtherConfig:
    """其他配置"""
    # 是否使用 DeepSpeed（需要安装 deepspeed）
    use_deepspeed: bool = False
    # DeepSpeed 配置文件路径
    deepspeed_config: Optional[str] = None
    # 是否启用 wandb 日志
    use_wandb: bool = False
    # wandb 项目名称
    wandb_project: str = ""
    # wandb 运行名称
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "OtherConfig":
        """从字典创建配置对象"""
        return cls(
            use_deepspeed=config.get("use_deepspeed", False),
            deepspeed_config=config.get("deepspeed_config"),
            use_wandb=config.get("use_wandb", False),
            wandb_project=config.get("wandb_project", ""),
            wandb_run_name=config.get("wandb_run_name"),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "use_deepspeed": self.use_deepspeed,
            "deepspeed_config": self.deepspeed_config,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
        }

