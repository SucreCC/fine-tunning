"""
模型训练主脚本
从 config.yaml 读取配置并执行训练
"""
import os
import yaml
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from bitsandbytes import BitsAndBytesConfig
from dataset import ConversationDataset


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config: dict):
    """加载模型和分词器"""
    model_config = config["model"]
    base_model_path = model_config["base_model_path"]
    
    # 加载分词器
    print(f"加载分词器: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 配置量化
    quantization_config = None
    if model_config.get("use_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    elif model_config.get("use_8bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # 加载模型
    print(f"加载模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if not quantization_config else None
    )
    
    # 如果使用量化，准备模型用于训练
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def setup_lora(model, config: dict):
    """设置 LoRA"""
    lora_config_dict = config.get("lora", {})
    
    if not lora_config_dict.get("use_lora", False):
        return model
    
    print("配置 LoRA...")
    
    # 确定任务类型
    task_type = TaskType.CAUSAL_LM
    
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=task_type,
        r=lora_config_dict.get("r", 8),
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        target_modules=lora_config_dict.get("target_modules", ["query_key_value"]),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.1),
        bias="none"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def create_datasets(config: dict, tokenizer):
    """创建训练和验证数据集"""
    script_dir = Path(__file__).parent.absolute()
    dataset_config = config["dataset"]
    
    # 处理训练集路径
    train_path = dataset_config["train_path"]
    if not os.path.isabs(train_path):
        train_path = str(script_dir.parent / train_path)
    
    train_dataset = ConversationDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=dataset_config.get("max_length", 2048)
    )
    
    # 处理验证集路径
    val_dataset = None
    if dataset_config.get("val_path"):
        val_path = dataset_config["val_path"]
        if not os.path.isabs(val_path):
            val_path = str(script_dir.parent / val_path)
        
        if os.path.exists(val_path):
            val_dataset = ConversationDataset(
                data_path=val_path,
                tokenizer=tokenizer,
                max_length=dataset_config.get("max_length", 2048)
            )
    
    return train_dataset, val_dataset


def train(config_path: str = "config.yaml"):
    """主训练函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    
    # 加载配置
    print("加载配置文件...")
    config_path_abs = script_dir / config_path if not os.path.isabs(config_path) else Path(config_path)
    config = load_config(str(config_path_abs))
    
    # 设置输出目录（相对于脚本目录）
    output_dir_str = config["model"]["output_dir"]
    if os.path.isabs(output_dir_str):
        output_dir = Path(output_dir_str)
    else:
        output_dir = script_dir.parent / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 设置 LoRA（如果启用）
    model = setup_lora(model, config)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    
    print(f"训练集大小: {len(train_dataset)}")
    if val_dataset:
        print(f"验证集大小: {len(val_dataset)}")
    
    # 训练参数
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2.0e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_config.get("warmup_steps", 100),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500) if val_dataset else None,
        logging_steps=training_config.get("logging_steps", 50),
        fp16=training_config.get("fp16", True),
        bf16=training_config.get("bf16", False),
        seed=training_config.get("seed", 42),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        report_to="wandb" if config.get("other", {}).get("use_wandb", False) else None,
        run_name=config.get("other", {}).get("wandb_project", "mu_xue_finetuning") if config.get("other", {}).get("use_wandb", False) else None,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型，不是掩码语言模型
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存模型...")
    if config.get("lora", {}).get("use_lora", False):
        # LoRA 模型保存
        model.save_pretrained(str(output_dir / "lora_model"))
        tokenizer.save_pretrained(str(output_dir / "lora_model"))
    else:
        # 全量模型保存
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    print("训练完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）"
    )
    
    args = parser.parse_args()
    
    train(args.config)

