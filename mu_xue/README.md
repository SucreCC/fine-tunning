# 沐雪模型训练

这是一个用于微调对话模型的训练框架，支持从 `config.yaml` 读取配置并执行训练。

## 功能特性

- ✅ 支持从 YAML 配置文件读取所有训练参数
- ✅ 支持 LoRA 微调（参数高效微调）
- ✅ 支持 4bit/8bit 量化训练（节省显存）
- ✅ 自动处理 JSONL 格式的对话数据
- ✅ 支持多种模型格式（ChatGLM、Qwen 等）
- ✅ 支持验证集评估
- ✅ 支持 wandb 实验跟踪（可选）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置文件说明

编辑 `config.yaml` 文件来配置训练参数：

### 模型配置
- `base_model_path`: 基础模型路径（HuggingFace 模型名称或本地路径）
- `output_dir`: 模型保存路径
- `use_8bit` / `use_4bit`: 是否使用量化（节省显存）

### 数据集配置
- `train_path`: 训练集 JSONL 文件路径
- `val_path`: 验证集 JSONL 文件路径
- `max_length`: 最大序列长度

### 训练配置
- `num_epochs`: 训练轮数
- `per_device_train_batch_size`: 批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率
- 等等...

### LoRA 配置
- `use_lora`: 是否使用 LoRA 微调
- `r`: LoRA rank
- `lora_alpha`: LoRA alpha
- `target_modules`: LoRA 目标模块（根据模型类型调整）

## 使用方法

### 基本使用

```bash
cd mu_xue
python train.py
```

### 指定配置文件

```bash
python train.py --config config.yaml
```

## 数据格式

训练数据应为 JSONL 格式，每行一个 JSON 对象：

```json
{
  "system": "你是一个名为沐雪的可爱AI女孩子",
  "conversation": [
    {
      "human": "你好",
      "assistant": "你好！我是沐雪，很高兴认识你~"
    },
    {
      "human": "今天天气怎么样？",
      "assistant": "我这边看不到天气呢，你可以看看窗外或者查一下天气预报哦~"
    }
  ]
}
```

## 模型保存

训练完成后，模型会保存在配置文件中指定的 `output_dir` 目录：

- 如果使用 LoRA：保存为 `lora_model/` 目录
- 如果全量微调：保存为 `final_model/` 目录

## 注意事项

1. **显存要求**：
   - 全量微调：需要较大显存（取决于模型大小）
   - LoRA 微调：显存需求大幅降低
   - 4bit 量化：进一步降低显存需求

2. **模型适配**：
   - 不同模型的 `target_modules` 可能不同
   - ChatGLM 模型：`["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]`
   - Qwen 模型：`["q_proj", "k_proj", "v_proj", "o_proj"]`
   - LLaMA 模型：`["q_proj", "k_proj", "v_proj", "o_proj"]`

3. **路径配置**：
   - 配置文件中的路径可以使用相对路径（相对于 `mu_xue` 目录）
   - 或使用绝对路径

## 示例配置

```yaml
model:
  base_model_path: "THUDM/chatglm3-6b"
  output_dir: "../model"
  use_4bit: true

dataset:
  train_path: "../dataset/moemuu/train.jsonl"
  val_path: "../dataset/moemuu/test.jsonl"
  max_length: 2048

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5

lora:
  use_lora: true
  r: 8
  lora_alpha: 32
  target_modules: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
```

