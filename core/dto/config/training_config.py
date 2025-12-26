"""
训练配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrecisionConfig:
    """精度配置"""
    fp16: bool = True
    bf16: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> "PrecisionConfig":
        """从字典创建配置对象"""
        return cls(
            fp16=config.get("fp16", True),
            bf16=config.get("bf16", False),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "fp16": self.fp16,
            "bf16": self.bf16,
        }


@dataclass
class DeviceConfig:
    """设备配置"""
    # 设备类型（"cpu" | "gpu" | "tpu" | None）
    # - "cpu": 强制使用 CPU
    # - "gpu": 强制使用 GPU（CUDA）
    # - "tpu": 强制使用 TPU（需要额外配置）
    # - None: 自动检测（优先使用 GPU，如果没有则使用 CPU）
    device_type: Optional[str] = None
    # 并行策略（"ddp" | "deepspeed" | "fsdp" | null）
    # - "ddp": DistributedDataParallel，单机多卡或多机多卡
    # - "deepspeed": DeepSpeed ZeRO，需要 deepspeed 配置文件
    # - "fsdp": Fully Sharded Data Parallel，PyTorch FSDP
    # - null: 不使用并行（单卡训练）
    parallel_strategy: Optional[str] = None
    # 分布式训练后端（"nccl" 用于多 GPU，"gloo" 用于 CPU，None 表示自动选择）
    # 当 parallel_strategy="ddp" 时使用
    ddp_backend: Optional[str] = None
    # DDP 是否查找未使用的参数（用于调试）
    ddp_find_unused_parameters: bool = False
    # DDP 超时时间（秒）
    ddp_timeout: int = 1800
    # 本地 rank（通常从环境变量获取，这里可以手动指定）
    local_rank: Optional[int] = None
    # DeepSpeed 配置文件路径（当 parallel_strategy="deepspeed" 时使用）
    deepspeed_config: Optional[str] = None
    # FSDP 配置（当 parallel_strategy="fsdp" 时使用）
    fsdp_config: Optional[dict] = None

    @classmethod
    def from_dict(cls, config: dict) -> "DeviceConfig":
        """从字典创建配置对象"""
        # 向后兼容：如果存在 use_cuda 或 use_cpu，转换为 device_type
        device_type = config.get("device_type", None)
        if device_type is None:
            # 兼容旧配置
            use_cuda = config.get("use_cuda", None)
            use_cpu = config.get("use_cpu", None)
            if use_cpu is True:
                device_type = "cpu"
            elif use_cuda is False:
                device_type = "cpu"
            elif use_cuda is True:
                device_type = "gpu"
            # 如果都是 None，则 device_type 保持为 None（自动检测）
        
        return cls(
            device_type=device_type,
            parallel_strategy=config.get("parallel_strategy", None),
            ddp_backend=config.get("ddp_backend", None),
            ddp_find_unused_parameters=config.get("ddp_find_unused_parameters", False),
            ddp_timeout=config.get("ddp_timeout", 1800),
            local_rank=config.get("local_rank", None),
            deepspeed_config=config.get("deepspeed_config", None),
            fsdp_config=config.get("fsdp_config", None),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {}
        if self.device_type is not None:
            result["device_type"] = self.device_type
        if self.parallel_strategy is not None:
            result["parallel_strategy"] = self.parallel_strategy
        if self.ddp_backend is not None:
            result["ddp_backend"] = self.ddp_backend
        result["ddp_find_unused_parameters"] = self.ddp_find_unused_parameters
        result["ddp_timeout"] = self.ddp_timeout
        if self.local_rank is not None:
            result["local_rank"] = self.local_rank
        if self.deepspeed_config is not None:
            result["deepspeed_config"] = self.deepspeed_config
        if self.fsdp_config is not None:
            result["fsdp_config"] = self.fsdp_config
        return result


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
    # 随机种子
    seed: int = 42
    # 最大梯度范数（用于梯度裁剪）
    max_grad_norm: float = 1.0
    # 保存模型总数限制
    save_total_limit: int = 3
    # 精度配置
    precision: Optional[PrecisionConfig] = None
    # 向后兼容：是否使用混合精度训练
    fp16: bool = True
    # 向后兼容：是否使用 bf16（需要 A100 等支持）
    bf16: bool = False
    # 设备配置
    device: Optional[DeviceConfig] = None

    def __post_init__(self):
        """初始化默认值"""
        if self.precision is None:
            self.precision = PrecisionConfig(fp16=self.fp16, bf16=self.bf16)
        # 同步 precision 到 fp16/bf16（向后兼容）
        if self.precision:
            self.fp16 = self.precision.fp16
            self.bf16 = self.precision.bf16
        # 初始化设备配置
        if self.device is None:
            self.device = DeviceConfig()

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        """从字典创建配置对象"""
        # 处理 precision 配置（支持新旧两种格式）
        precision_config = config.get("precision", {})
        if isinstance(precision_config, dict) and precision_config:
            precision = PrecisionConfig.from_dict(precision_config)
            fp16 = precision.fp16
            bf16 = precision.bf16
        else:
            # 向后兼容：从顶层读取
            fp16 = config.get("fp16", True)
            bf16 = config.get("bf16", False)
            precision = PrecisionConfig(fp16=fp16, bf16=bf16)
        
        # 处理 device 配置
        device_config = config.get("device", {})
        if isinstance(device_config, dict) and device_config:
            device = DeviceConfig.from_dict(device_config)
        else:
            device = DeviceConfig()
        
        instance = cls(
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
            seed=config.get("seed", 42),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            save_total_limit=config.get("save_total_limit", 3),
            precision=precision,
            fp16=fp16,
            bf16=bf16,
            device=device,
        )
        return instance

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
            "seed": self.seed,
            "max_grad_norm": self.max_grad_norm,
            "save_total_limit": self.save_total_limit,
            "precision": self.precision.to_dict() if self.precision else PrecisionConfig().to_dict(),
            "device": self.device.to_dict() if self.device else DeviceConfig().to_dict(),
        }
