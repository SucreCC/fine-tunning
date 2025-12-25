"""
模型加载模块
加载 base model 和 tokenizer
"""
import torch
from typing import Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


def load_model_and_tokenizer(
    model_config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载模型和分词器

    Args:
        model_config: 模型配置

    Returns:
        (model, tokenizer)
    """
    base_model_path = model_config.base_model_path

    # 1. 加载 tokenizer
    logger.info(f"加载分词器: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # CausalLM 通常必须有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    if model_config.quantization and model_config.quantization.enable:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                "启用了量化，但 transformers 中未找到 BitsAndBytesConfig。"
                "请安装 bitsandbytes: pip install bitsandbytes"
            ) from e

        # 根据配置的 bits 和 compute_dtype 设置量化
        if model_config.quantization.bits == 4:
            # 确定计算数据类型
            compute_dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            compute_dtype = compute_dtype_map.get(
                model_config.quantization.compute_dtype,
                torch.bfloat16
            )
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif model_config.quantization.bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            raise ValueError(
                f"不支持的量化位数: {model_config.quantization.bits}。"
                f"支持的位数: 4 或 8"
            )

    # 3. 加载模型
    logger.info(f"加载模型: {base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=None if quantization_config else torch.float16,
    )

    return model, tokenizer