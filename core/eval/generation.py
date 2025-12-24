"""
生成评估模块
定点 prompt 生成评估
"""
import torch
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    生成文本
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示词
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: top-p 采样参数
        do_sample: 是否使用采样
        
    Returns:
        生成的文本
    """
    model.eval()
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除输入部分
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "").strip()
    
    return generated_text


def evaluate_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_prompts: List[str],
    **generation_kwargs
) -> List[Dict[str, str]]:
    """
    评估生成结果
    
    Args:
        model: 模型
        tokenizer: 分词器
        test_prompts: 测试提示词列表
        **generation_kwargs: 生成参数
        
    Returns:
        评估结果列表，每个元素包含 prompt 和 generated_text
    """
    results = []
    
    for prompt in test_prompts:
        generated_text = generate_text(model, tokenizer, prompt, **generation_kwargs)
        results.append({
            "prompt": prompt,
            "generated_text": generated_text
        })
    
    return results

