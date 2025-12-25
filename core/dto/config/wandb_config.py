"""
Wandb 配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class WandbConfig:
    """Wandb 配置"""
    # 是否启用 wandb 日志
    use_wandb: bool = False
    # wandb API Key（可选，如果不设置则使用环境变量 WANDB_API_KEY 或已登录的 key）
    # 注意：建议使用环境变量或 wandb login 命令，而不是在配置文件中硬编码
    wandb_api_key: Optional[str] = None
    # wandb 项目名称
    wandb_project: str = ""
    # wandb 运行名称
    wandb_run_name: Optional[str] = None
    # wandb 实体（组织或个人账户）
    wandb_entity: Optional[str] = None
    # wandb 标签
    wandb_tags: Optional[list] = None
    # wandb 保存目录
    wandb_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "WandbConfig":
        """从字典创建配置对象"""
        return cls(
            use_wandb=config.get("use_wandb", False),
            wandb_api_key=config.get("wandb_api_key"),
            wandb_project=config.get("wandb_project", ""),
            wandb_run_name=config.get("wandb_run_name"),
            wandb_entity=config.get("wandb_entity"),
            wandb_tags=config.get("wandb_tags"),
            wandb_dir=config.get("wandb_dir"),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "use_wandb": self.use_wandb,
            "wandb_api_key": self.wandb_api_key,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "wandb_entity": self.wandb_entity,
            "wandb_tags": self.wandb_tags,
            "wandb_dir": self.wandb_dir,
        }

