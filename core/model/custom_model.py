"""
模型加载模块
封装模型和分词器的加载逻辑
"""
import torch
from typing import Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomModel:
    """自定义模型类，封装模型和分词器的加载逻辑"""
    
    def __init__(self, model_config: ModelConfig):
        """
        初始化 CustomModel
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._quantization_config = None
    
    def _setup_quantization(self) -> Optional[object]:
        """
        设置量化配置
        
        Returns:
            BitsAndBytesConfig 对象，如果未启用量化则返回 None
        """
        if not (self.model_config.quantization and self.model_config.quantization.enable):
            return None
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                "启用了量化，但 transformers 中未找到 BitsAndBytesConfig。"
                "请安装 bitsandbytes: pip install bitsandbytes"
            ) from e

        # 根据配置的 bits 和 compute_dtype 设置量化
        if self.model_config.quantization.bits == 4:
            # 确定计算数据类型
            compute_dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            compute_dtype = compute_dtype_map.get(
                self.model_config.quantization.compute_dtype,
                torch.bfloat16
            )
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.model_config.quantization.bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            raise ValueError(
                f"不支持的量化位数: {self.model_config.quantization.bits}。"
                f"支持的位数: 4 或 8"
            )
        
        return quantization_config
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        加载分词器
        
        Returns:
            分词器对象
        """
        base_model_path = self.model_config.base_model_path
        
        logger.info(f"加载分词器: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

        # CausalLM 通常必须有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer
    
    def _load_model(self) -> PreTrainedModel:
        """
        加载模型
        
        Returns:
            模型对象
        """
        base_model_path = self.model_config.base_model_path
        
        logger.info(f"加载模型: {base_model_path}")
        
        # 设置量化配置
        quantization_config = self._setup_quantization()
        self._quantization_config = quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=None if quantization_config else torch.float16,
        )
        
        return model
    
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        加载模型和分词器
        
        Returns:
            (model, tokenizer) 元组
        """
        # 加载分词器
        self.tokenizer = self._load_tokenizer()
        
        # 加载模型
        self.model = self._load_model()
        
        logger.info(f"模型和分词器加载成功: {self.model_config.base_model_path}")
        
        return self.model, self.tokenizer
    
    def save(self, save_path: str):
        """
        保存模型和分词器
        
        Args:
            save_path: 保存路径
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型和分词器尚未加载，请先调用 load() 方法")
        
        logger.info(f"保存模型和分词器到: {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("保存完成")


# 为了向后兼容，保留函数接口
def load_model_and_tokenizer(
    model_config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载模型和分词器（向后兼容函数）
    
    Args:
        model_config: 模型配置
    
    Returns:
        (model, tokenizer) 元组
    """
    custom_model = CustomModel(model_config)
    return custom_model.load()
