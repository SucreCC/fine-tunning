"""
训练入口脚本
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.config import load_config
from core.utils.seed import set_seed
from core.model import load_model_and_tokenizer, setup_lora
from core.data import ConversationDataset
from core.training import create_trainer, get_callbacks


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print("加载配置文件...")
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config.training_config.seed)
    
    # 设置输出目录
    output_dir = Path(config.model_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和分词器
    print("加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 设置 LoRA（如果启用）
    model = setup_lora(model, config)
    
    # 创建数据集
    print("创建数据集...")
    # 处理路径
    script_dir = Path(__file__).parent.parent
    train_path = config.dataset_config.train_path
    if not os.path.isabs(train_path):
        train_path = str(script_dir / train_path)
    
    train_dataset = ConversationDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=config.dataset_config.max_length
    )
    
    val_dataset = None
    if config.dataset_config.val_path:
        val_path = config.dataset_config.val_path
        if not os.path.isabs(val_path):
            val_path = str(script_dir / val_path)
        
        if os.path.exists(val_path):
            val_dataset = ConversationDataset(
                data_path=val_path,
                tokenizer=tokenizer,
                max_length=config.dataset_config.max_length
            )
    
    print(f"训练集大小: {len(train_dataset)}")
    if val_dataset:
        print(f"验证集大小: {len(val_dataset)}")
    
    # 创建 Trainer
    print("创建 Trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config
    )
    
    # 添加回调
    callbacks = get_callbacks(config)
    for callback in callbacks:
        trainer.add_callback(callback)
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存模型...")
    if config.lora_config.use_lora:
        # LoRA 模型保存
        model.save_pretrained(str(output_dir / "lora_model"))
        tokenizer.save_pretrained(str(output_dir / "lora_model"))
    else:
        # 全量模型保存
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    print("训练完成！")


if __name__ == "__main__":
    main()

