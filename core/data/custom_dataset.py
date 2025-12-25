"""
数据集处理模块
Dataset 和 DataCollator
"""
import json
import random
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling

from core.data.interface.base_process import BaseProcess
from core.utils import logging

logger = logging.get_logger(__name__)

class CustomDataset(Dataset):
    """对话数据集类"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        process: BaseProcess,
        max_length: int = 2048,
        train_ratio: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            process: 数据处理对象，用于格式化对话
            max_length: 最大序列长度
            train_ratio: 训练集使用比例（0.0-1.0，1.0 表示使用全部数据）
            seed: 随机种子，用于数据子集选择的可重复性
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process = process
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """
        加载 JSONL 数据，并根据 train_ratio 选择子集
        
        Args:
            data_path: JSONL 数据文件路径
            
        Returns:
            数据列表（可能根据 train_ratio 进行了子集选择）
        """
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.info(f"解析 JSON 失败: {line[:100]}... 错误: {e}")
                        continue
        
        # 根据 train_ratio 选择子集
        if self.train_ratio < 1.0:
            total_size = len(data)
            subset_size = int(total_size * self.train_ratio)
            
            # 设置随机种子以确保可重复性
            if self.seed is not None:
                random.seed(self.seed)
            
            # 随机打乱并选择子集
            indices = list(range(total_size))
            random.shuffle(indices)
            selected_indices = sorted(indices[:subset_size])  # 排序以保持原始顺序
            
            data = [data[i] for i in selected_indices]
            logger.info(
                f"使用数据集的 {self.train_ratio*100:.1f}%: "
                f"从 {total_size} 条数据中选择 {subset_size} 条"
            )
        
        return data
    
    def _format_conversation(self, item: Dict) -> str:
        """
        格式化对话为模型输入格式
        
        Args:
            item: 包含 system 和 conversation 的字典
            
        Returns:
            格式化后的对话字符串
        """
        return self.process.process(item)
    
    def _tokenize(self, text: str) -> Dict:
        """
        对文本进行分词和编码
        
        Args:
            text: 输入文本
            
        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        # 使用 tokenizer 编码
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个数据样本"""
        item = self.data[idx]
        
        # 格式化对话
        formatted_text = self._format_conversation(item)
        
        # 分词编码
        encoded = self._tokenize(formatted_text)
        
        # 创建 labels（对于生成任务，labels 通常等于 input_ids）
        # 但需要将 padding 部分设为 -100（忽略损失计算）
        labels = encoded["input_ids"].copy()
        attention_mask = encoded["attention_mask"]
        
        # 将 padding 部分的 labels 设为 -100
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_data_collator(tokenizer: PreTrainedTokenizer):
    """
    获取数据整理器
    
    Args:
        tokenizer: 分词器
        
    Returns:
        DataCollatorForLanguageModeling 实例
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型，不是掩码语言模型
    )

