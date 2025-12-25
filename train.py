"""
训练入口脚本
"""
import os
from pathlib import Path

from core.config.config_manager import ConfigManager
from core.config.log_config import LogConfig
from core.config.model_config import ModelConfig
from core.data import ConversationDataset
from core.model import load_model_and_tokenizer, setup_lora
from core.training import create_trainer, get_callbacks
from core.utils import logging
from core.utils.file_utils import find_project_root
from core.utils.logging import setup_logging
from core.utils.seed import set_seed

PROJECT_ROOT =find_project_root()


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


def main():
    """主训练函数"""
    config_manager = init_config()
    logger = init_log(config_manager.log_config)

    # 设置随机种子
    set_seed(config_manager.training_config.seed)
    logger.info(f"随机种子: {config_manager.training_config.seed}")

    output_dir = init_output_dir(config_manager.model_config)
    logger.info(f"模型输出目录: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(config_manager)


    # load_model_and_tokenizer(config_manager.model_config)

    # # 加载模型和分词器
    # print("加载模型和分词器...")

    #
    # # 设置 LoRA（如果启用）
    # model = setup_lora(model, config)
    #
    # # 创建数据集
    # print("创建数据集...")
    # # 处理路径
    # script_dir = Path(__file__).parent.parent
    # train_path = config.dataset_config.train_path
    # if not os.path.isabs(train_path):
    #     train_path = str(script_dir / train_path)
    #
    # train_dataset = ConversationDataset(
    #     data_path=train_path,
    #     tokenizer=tokenizer,
    #     max_length=config.dataset_config.max_length
    # )
    #
    # val_dataset = None
    # if config.dataset_config.val_path:
    #     val_path = config.dataset_config.val_path
    #     if not os.path.isabs(val_path):
    #         val_path = str(script_dir / val_path)
    #
    #     if os.path.exists(val_path):
    #         val_dataset = ConversationDataset(
    #             data_path=val_path,
    #             tokenizer=tokenizer,
    #             max_length=config.dataset_config.max_length
    #         )
    #
    # print(f"训练集大小: {len(train_dataset)}")
    # if val_dataset:
    #     print(f"验证集大小: {len(val_dataset)}")
    #
    # # 创建 Trainer
    # print("创建 Trainer...")
    # trainer = create_trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     config=config
    # )
    #
    # # 添加回调
    # callbacks = get_callbacks(config)
    # for callback in callbacks:
    #     trainer.add_callback(callback)
    #
    # # 开始训练
    # print("开始训练...")
    # trainer.train()
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
