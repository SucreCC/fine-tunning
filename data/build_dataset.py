"""
数据构建脚本
从原始数据构建训练数据集
"""
import json
import os
from pathlib import Path
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败: {line[:100]}... 错误: {e}")
                    continue
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    保存数据到 JSONL 文件
    
    Args:
        data: 数据列表
        file_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def clean_data(data: List[Dict]) -> List[Dict]:
    """
    清洗数据
    
    Args:
        data: 原始数据
        
    Returns:
        清洗后的数据
    """
    cleaned_data = []
    
    for item in data:
        # 检查必需字段
        if "conversation" not in item:
            continue
        
        # 检查对话是否为空
        if not item.get("conversation"):
            continue
        
        # 检查每个对话是否有 human 和 assistant
        valid = True
        for conv in item["conversation"]:
            if "human" not in conv or "assistant" not in conv:
                valid = False
                break
        
        if valid:
            cleaned_data.append(item)
    
    return cleaned_data


def split_dataset(data: List[Dict], train_ratio: float = 0.9) -> tuple:
    """
    划分训练集和验证集
    
    Args:
        data: 数据列表
        train_ratio: 训练集比例
        
    Returns:
        (train_data, val_data) 元组
    """
    import random
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据构建脚本")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="原始数据目录（默认: data/raw）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="处理后数据输出目录（默认: data/processed）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例（默认: 0.9）"
    )
    
    args = parser.parse_args()
    
    # 加载所有原始数据
    print("加载原始数据...")
    input_dir = Path(args.input_dir)
    all_data = []
    
    for jsonl_file in input_dir.glob("*.jsonl"):
        print(f"加载文件: {jsonl_file}")
        data = load_jsonl(str(jsonl_file))
        all_data.extend(data)
        print(f"  加载了 {len(data)} 条数据")
    
    print(f"总共加载了 {len(all_data)} 条数据")
    
    # 清洗数据
    print("\n清洗数据...")
    cleaned_data = clean_data(all_data)
    print(f"清洗后剩余 {len(cleaned_data)} 条数据")
    
    # 划分训练集和验证集
    print("\n划分数据集...")
    train_data, val_data = split_dataset(cleaned_data, args.train_ratio)
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(train_data, str(output_dir / "train.jsonl"))
    save_jsonl(val_data, str(output_dir / "val.jsonl"))
    
    print(f"\n数据构建完成！")
    print(f"训练集: {output_dir / 'train.jsonl'}")
    print(f"验证集: {output_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()

