"""
数据集处理模块
Dataset 和 DataCollator
"""
import json
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling


class ConversationDataset(Dataset):
    """对话数据集类"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        system_template: str = "你是一个名为沐雪的可爱AI女孩子"
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            system_template: 系统提示模板
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_template = system_template
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载 JSONL 数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
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
    
    def _format_conversation(self, item: Dict) -> str:
        """
        格式化对话为模型输入格式
        
        Args:
            item: 包含 system 和 conversation 的字典
            
        Returns:
            格式化后的对话字符串
        """
        from .preprocess import format_conversation
        return format_conversation(item, self.system_template)
    
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

