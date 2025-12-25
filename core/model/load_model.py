"""
模型加载模块
加载 base model 和 tokenizer
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Any, TYPE_CHECKING
from transformers.models.auto.modeling_auto import _BaseModelWithGenerate
from core.config.model_config import ModelConfig
from core.utils import logging

try:
    from bitsandbytes import BitsAndBytesConfig
except ImportError:
    # 如果未安装 bitsandbytes，设置为 None
    BitsAndBytesConfig = None  # type: ignore

logger = logging.get_logger(__name__)


def load_model_and_tokenizer(model_config: ModelConfig) -> tuple[_BaseModelWithGenerate, Any]:
    """
    加载模型和分词器
    
    Args:
        config: 配置管理器
        
    Returns:
        (model, tokenizer) 元组
    """
    base_model_path = model_config.base_model_path
    
    # 加载分词器
    logger.info(f"加载分词器: {base_model_path}")
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
    if model_config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    elif model_config.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # 加载模型
    logger.info(f"加载模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if not quantization_config else None
    )
    
    return model, tokenizer

