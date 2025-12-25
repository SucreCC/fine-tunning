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
from core.config.model_config import ModelConfig
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
    if model_config.use_4bit or model_config.use_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                "启用了 4bit/8bit 量化，但 transformers 中未找到 BitsAndBytesConfig"
            ) from e

        if model_config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
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