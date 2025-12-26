"""
训练入口脚本
"""
import os

# 设置 tokenizers 并行性环境变量，避免 fork 警告
# 必须在导入 transformers 之前设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import PreTrainedTokenizer
from core.data.custom_dataset import CustomDataset
from core.data.interface.base_processor import BaseProcessor
from core.dto.config.config_manager import ConfigManager
from core.dto.config.log_config import LogConfig
from core.dto.config.model_config import ModelConfig
from core.model.custom_model import CustomModel
from core.training import CustomTrainer
from core.utils import logging
from core.utils.file_utils import find_project_root
from core.utils.logging import setup_logging
from core.utils.seed import set_seed

PROJECT_ROOT = find_project_root()


def init_config():
    """初始化配置"""
    config_manager = ConfigManager()
    return config_manager.from_yaml()


def init_log(log_config: LogConfig):
    """初始化日志"""
    setup_logging(log_config)
    return logging.get_logger(__name__)


def init_output_dir(model_config: ModelConfig):
    """初始化输出目录"""
    output_dir_path = model_config.get_output_dir()
    output_dir = PROJECT_ROOT / output_dir_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def init_data_set(config_manager: ConfigManager, tokenizer: PreTrainedTokenizer):
    """初始化数据集"""

    # 根据配置创建数据处理对象
    processor = BaseProcessor.get_processor(config_manager.dataset_config)

    # 初始化训练数据集
    train_dataset = CustomDataset(
        data_path=config_manager.dataset_config.train_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=config_manager.dataset_config.max_length,
        train_ratio=config_manager.dataset_config.train_ratio,
        seed=config_manager.training_config.seed
    )

    # 初始化验证数据集
    val_dataset = CustomDataset(
        data_path=config_manager.dataset_config.val_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=config_manager.dataset_config.max_length,
        train_ratio=1.0,  # 验证集使用全部数据
        seed=None  # 验证集不需要随机种子
    )

    return train_dataset, val_dataset


def main():
    """主训练函数"""
    config_manager = init_config()
    logger = init_log(config_manager.log_config)

    # 设置随机种子
    set_seed(config_manager.training_config.seed)
    logger.info(f"随机种子: {config_manager.training_config.seed}")

    # 创建输出目录
    output_dir = init_output_dir(config_manager.model_config)
    logger.info(f"模型输出目录: {output_dir}")

    # 初始化模型和分词器
    model, tokenizer = CustomModel(
        model_config=config_manager.model_config,
        finetune_config=config_manager.finetune_config,
    ).load()

    train_dataset, val_dataset = init_data_set(config_manager, tokenizer)
    
    # 检查数据集
    if len(train_dataset) == 0:
        logger.error("训练数据集为空！请检查数据集路径和 train_ratio 配置")
        raise ValueError("训练数据集为空，无法开始训练")

    # 创建 Trainer（回调会自动添加）
    logger.info("正在创建 Trainer...")
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config_manager
    )
    logger.info("✓ Trainer 创建成功")
    
    # 输出训练信息
    logger.info("=" * 60)
    logger.info("训练配置信息:")
    logger.info(f"  训练数据集大小: {len(train_dataset)}")
    logger.info(f"  验证数据集大小: {len(val_dataset) if val_dataset else 0}")
    logger.info(f"  训练轮数: {config_manager.training_config.num_epochs}")
    logger.info(f"  批次大小: {config_manager.training_config.per_device_train_batch_size}")
    logger.info(f"  梯度累积步数: {config_manager.training_config.gradient_accumulation_steps}")
    effective_batch_size = config_manager.training_config.per_device_train_batch_size * config_manager.training_config.gradient_accumulation_steps
    logger.info(f"  有效批次大小: {effective_batch_size}")
    logger.info(f"  学习率: {config_manager.training_config.learning_rate}")
    logger.info(f"  日志步数间隔: {config_manager.training_config.logging_steps}")
    logger.info(f"  保存步数间隔: {config_manager.training_config.save_steps}")
    
    # 计算总步数
    total_steps = (len(train_dataset) // effective_batch_size) * config_manager.training_config.num_epochs
    logger.info(f"  预计总训练步数: {total_steps}")
    logger.info("=" * 60)

    # 开始训练
    logger.info("")
    logger.info("🚀 开始训练...")
    logger.info("=" * 60)
    try:
        trainer.train()
        logger.info("=" * 60)
        logger.info("✓ 训练完成！")
    except KeyboardInterrupt:
        logger.info("=" * 60)
        logger.warning("训练被用户中断")
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise

    # 保存最终模型
    logger.info("保存模型...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
