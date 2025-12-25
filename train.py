"""
训练入口脚本
"""
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
    output_dir = PROJECT_ROOT / model_config.output_dir
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




    # 创建 Trainer（回调会自动添加）
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config_manager
    )
    logger.info("创建 Trainer 成功")

    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    #
    # # 保存最终模型
    # print("保存模型...")
    # if config.lora_config.use_lora:
    #     # LoRA 模型保存
    #     model.save_pretrained(str(output_dir / "lora_model"))
    #     tokenizer.save_pretrained(str(output_dir / "lora_model"))
    # else:
    #     # 全量模型保存
    #     trainer.save_model(str(output_dir / "final_model"))
    #     tokenizer.save_pretrained(str(output_dir / "final_model"))
    #
    # print("训练完成！")


if __name__ == "__main__":
    main()
