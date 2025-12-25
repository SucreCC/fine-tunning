"""
数据集配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 训练集路径
    train_path: str = ""
    # 验证集路径
    val_path: Optional[str] = ""
    # 最大序列长度
    max_length: int = 2048
    # 是否使用流式加载（节省内存）
    streaming: bool = False
    # 训练集使用百分比（0.0-1.0，1.0 表示使用全部数据）
    train_ratio: float = 1.0

    # 数据处理器配置
    # processor 可以是字符串（向后兼容）或字典（新格式）
    processor: Optional[str] = None
    # 系统提示词（从 processor.system_prompt 或直接配置）
    system_prompt: str = ""

    @classmethod
    def from_dict(cls, config: dict) -> "DatasetConfig":
        """从字典创建配置对象"""
        # 处理 processor 配置（支持新旧两种格式）
        processor_config = config.get("processor", {})
        processor_name = None
        system_prompt = ""
        
        if isinstance(processor_config, str):
            # 旧格式：processor: "qwen_processor"
            processor_name = processor_config
            system_prompt = config.get("system_prompt", "")
        elif isinstance(processor_config, dict):
            # 新格式：processor: {type: "qwen", system_prompt: "..."}
            processor_type = processor_config.get("type", "default")
            processor_name = f"{processor_type}_processor"
            system_prompt = processor_config.get("system_prompt", "")
        else:
            # 默认值
            processor_name = "default_processor"
            system_prompt = config.get("system_prompt", "")
        
        return cls(
            train_path=config.get("train_path", ""),
            val_path=config.get("val_path", ""),
            max_length=config.get("max_length", 2048),
            streaming=config.get("streaming", False),
            train_ratio=config.get("train_ratio", 1.0),
            processor=processor_name,
            system_prompt=system_prompt,
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        # 使用新格式输出
        return {
            "train_path": self.train_path,
            "val_path": self.val_path,
            "max_length": self.max_length,
            "streaming": self.streaming,
            "train_ratio": self.train_ratio,
            "processor": {
                "type": self.processor.replace("_processor", "") if self.processor else "default",
                "system_prompt": self.system_prompt,
            },
        }

