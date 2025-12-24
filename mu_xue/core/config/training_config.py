"""
训练配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练轮数
    num_epochs: int = 3
    # 批次大小
    per_device_train_batch_size: int = 2
    # 梯度累积步数
    gradient_accumulation_steps: int = 8
    # 学习率
    learning_rate: float = 2.0e-5
    # 权重衰减
    weight_decay: float = 0.01
    # 学习率调度器类型
    lr_scheduler_type: str = "cosine"
    # 预热步数
    warmup_steps: int = 100
    # 保存步数间隔
    save_steps: int = 500
    # 评估步数间隔
    eval_steps: int = 500
    # 日志记录步数间隔
    logging_steps: int = 50
    # 是否使用混合精度训练
    fp16: bool = True
    # 是否使用 bf16（需要 A100 等支持）
    bf16: bool = False
    # 随机种子
    seed: int = 42
    # 最大梯度范数（用于梯度裁剪）
    max_grad_norm: float = 1.0
    # 保存模型总数限制
    save_total_limit: int = 3

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        """从字典创建配置对象"""
        return cls(
            num_epochs=config.get("num_epochs", 3),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            learning_rate=config.get("learning_rate", 2.0e-5),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            warmup_steps=config.get("warmup_steps", 100),
            save_steps=config.get("save_steps", 500),
            eval_steps=config.get("eval_steps", 500),
            logging_steps=config.get("logging_steps", 50),
            fp16=config.get("fp16", True),
            bf16=config.get("bf16", False),
            seed=config.get("seed", 42),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            save_total_limit=config.get("save_total_limit", 3),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "num_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": self.warmup_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "seed": self.seed,
            "max_grad_norm": self.max_grad_norm,
            "save_total_limit": self.save_total_limit,
        }
