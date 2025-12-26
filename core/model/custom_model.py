"""
模型加载模块
封装模型和分词器的加载逻辑
"""
import os

# 设置 tokenizers 并行性环境变量，避免 fork 警告
# 必须在导入 transformers 之前设置
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from typing import Tuple, Optional, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftMixedModel

from core.dto.config.finetune_config.base_finetuning_config import BaseFinetuningConfig
from core.dto.config.model_config import ModelConfig
from core.utils import logging

logger = logging.get_logger(__name__)


class CustomModel:
    """自定义模型类，封装模型和分词器的加载逻辑"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        finetune_config: Optional[BaseFinetuningConfig] = None,
    ):
        """
        初始化 CustomModel
        
        Args:
            model_config: 模型配置
            finetune_config: 微调配置，如果提供且 enable 为 True，则会在加载模型后自动应用
        """
        self.model_config = model_config
        self.finetune_config = finetune_config
        self.model: Optional[Union[PreTrainedModel, PeftModel, PeftMixedModel]] = None
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
    
    def _sync_token_ids_to_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        同步 tokenizer 的 token IDs 到模型配置，避免警告
        
        Args:
            model: 模型对象
            tokenizer: 分词器对象
        """
        if not hasattr(model, 'config') or model.config is None:
            return
        
        config = model.config
        
        # 同步 pad_token_id
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        elif hasattr(config, 'pad_token_id') and config.pad_token_id is not None:
            # 如果 tokenizer 没有 pad_token_id，但模型配置有，保持模型配置
            pass
        
        # 同步 bos_token_id（包括 None 的情况）
        if hasattr(tokenizer, 'bos_token_id'):
            config.bos_token_id = tokenizer.bos_token_id
        
        # 同步 eos_token_id
        if tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id
        elif hasattr(config, 'eos_token_id') and config.eos_token_id is not None:
            # 如果 tokenizer 没有 eos_token_id，但模型配置有，保持模型配置
            pass
        
        # 同步 generation_config（如果存在）
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            gen_config = model.generation_config
            if tokenizer.pad_token_id is not None:
                gen_config.pad_token_id = tokenizer.pad_token_id
            if hasattr(tokenizer, 'bos_token_id'):
                gen_config.bos_token_id = tokenizer.bos_token_id
            if tokenizer.eos_token_id is not None:
                gen_config.eos_token_id = tokenizer.eos_token_id
    
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
    
    def _load_model(self, tokenizer: Optional[PreTrainedTokenizer] = None) -> PreTrainedModel:
        """
        加载模型
        
        Args:
            tokenizer: 分词器（可选），如果提供，会在加载模型时同步 token IDs
        
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
        
        # 如果提供了 tokenizer，立即同步 token IDs 以避免警告
        if tokenizer is not None:
            self._sync_token_ids_to_model(model, tokenizer)
        
        return model
    
    def _setup_finetuning(self) -> Union[PreTrainedModel, PeftModel, PeftMixedModel]:
        """
        设置微调策略
        
        Returns:
            应用了微调策略的模型
        """
        if self.model is None:
            raise ValueError("模型尚未加载，请先调用 _load_model() 方法")
        
        if self.finetune_config is None:
            logger.info("未提供微调配置，使用全量模型训练")
            return self.model
        
        if not self.finetune_config.enable:
            logger.info("微调已禁用，使用全量模型训练")
            return self.model
        
        # 动态获取微调实现类
        from core.dto.enums.finetuning_enum import FinetuneEnum
        
        finetune_type = self.finetune_config.type
        
        # 处理 full 策略（全量微调，不需要特殊处理）
        if finetune_type == "full":
            logger.info("使用全量微调，无需应用微调策略")
            return self.model
        
        finetune_class = FinetuneEnum.get_finetune_class_by_type(finetune_type)
        
        # 创建微调实例并应用
        finetune_instance = finetune_class()
        self.model = finetune_instance.setup(
            model=self.model,
            finetune_config=self.finetune_config,
            model_config=self.model_config,
        )
        
        logger.info(f"微调策略 '{finetune_type}' 设置成功")
        return self.model
    
    def load(self) -> Tuple[Union[PreTrainedModel, PeftModel, PeftMixedModel], PreTrainedTokenizer]:
        """
        加载模型和分词器，并应用微调策略（如果提供）
        
        Returns:
            (model, tokenizer) 元组
        """
        # 先加载分词器
        self.tokenizer = self._load_tokenizer()
        
        # 加载模型时传入 tokenizer，立即同步 token IDs 以避免警告
        self.model = self._load_model(tokenizer=self.tokenizer)
        
        logger.info(f"模型和分词器加载成功: {self.model_config.base_model_path}")
        
        # 如果提供了微调配置，应用微调策略
        if self.finetune_config is not None:
            self.model = self._setup_finetuning()
            # 应用微调后再次同步（某些微调方法可能会修改模型配置）
            self._sync_token_ids_to_model(self.model, self.tokenizer)
        
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

