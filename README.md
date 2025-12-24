# Fine-tuning 项目

模型微调项目，按照标准工程结构组织。

## 项目结构

```
fine-tuning/
├── configs/                # 所有可调参数
│   ├── config.yaml         # 主配置文件
│   ├── config_manager.py   # 配置管理器
│   └── ...                 # 其他配置类
│
├── data/
│   ├── raw/                # 原始数据（不可改）
│   ├── processed/          # 清洗后的数据
│   └── build_dataset.py    # 数据构建脚本
│
├── core/
│   ├── data/
│   │   ├── dataset.py      # Dataset / collator
│   │   └── preprocess.py   # 文本预处理
│   │
│   ├── model/
│   │   ├── load_model.py   # 加载 base model + tokenizer
│   │   └── lora.py         # LoRA / QLoRA 注入
│   │
│   ├── training/
│   │   ├── trainer.py      # Trainer 封装
│   │   └── callbacks.py    # logging / generation callback
│   │
│   ├── eval/
│   │   └── generation.py   # 定点 prompt 生成评估
│   │
│   └── utils/
│       ├── config.py        # 统一配置读取
│       └── seed.py          # 随机种子设置
│
├── scripts/
│   ├── train.py            # 训练入口
│   └── eval.py             # 推理 / 对比
│
└── outputs/
    ├── checkpoints/         # 模型检查点
    └── logs/                # 训练日志
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将原始数据放入 `data/raw/` 目录，然后运行数据构建脚本：

```bash
python data/build_dataset.py --input_dir data/raw --output_dir data/processed
```

### 3. 配置训练参数

编辑 `configs/config.yaml` 文件，设置模型路径、数据集路径、训练参数等。

### 4. 开始训练

```bash
python scripts/train.py --config configs/config.yaml
```

### 5. 评估模型

```bash
python scripts/eval.py --config configs/config.yaml --model_path outputs/checkpoints/lora_model
```

## 配置说明

所有配置都在 `configs/config.yaml` 中，包括：

- **model**: 模型配置（模型路径、输出目录、量化选项）
- **dataset**: 数据集配置（训练集/验证集路径、最大长度）
- **training**: 训练配置（学习率、批次大小、训练轮数等）
- **lora**: LoRA 配置（rank、alpha、target_modules 等）
- **wandb**: Wandb 配置（实验跟踪）
- **other**: 其他配置（DeepSpeed 等）

## 使用示例

### 训练

```bash
# 使用默认配置
python scripts/train.py

# 指定配置文件
python scripts/train.py --config configs/config.yaml
```

### 评估

```bash
# 使用默认配置和提示词
python scripts/eval.py

# 指定模型路径和自定义提示词
python scripts/eval.py \
    --model_path outputs/checkpoints/lora_model \
    --prompts "你好" "介绍一下你自己" "今天天气怎么样"
```

### 数据构建

```bash
# 使用默认路径
python data/build_dataset.py

# 指定输入输出目录
python data/build_dataset.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --train_ratio 0.9
```

## 注意事项

1. **路径配置**: 配置文件中的路径可以使用相对路径（相对于项目根目录）或绝对路径
2. **模型保存**: LoRA 模型会保存在 `outputs/checkpoints/lora_model/`，全量模型保存在 `outputs/checkpoints/final_model/`
3. **日志**: 训练日志会保存在 `outputs/logs/` 目录
4. **Wandb**: 如果启用 wandb，确保已安装并登录（`wandb login`）

