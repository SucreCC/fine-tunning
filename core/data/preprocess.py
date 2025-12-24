"""
文本预处理模块
"""
from typing import Dict, Optional
from transformers import PreTrainedTokenizer


def preprocess_text(text: str, max_length: Optional[int] = None) -> str:
    """
    预处理文本
    
    Args:
        text: 原始文本
        max_length: 最大长度（可选）
        
    Returns:
        预处理后的文本
    """
    # 去除首尾空白
    text = text.strip()
    
    # 可以在这里添加其他预处理逻辑
    # 例如：去除特殊字符、标准化等
    
    return text


def format_conversation(item: Dict, system_template: str = "你是一个名为沐雪的可爱AI女孩子", model_type: Optional[str] = None) -> str:
    """
    格式化对话为模型输入格式
    
    Args:
        item: 包含 system 和 conversation 的字典
        system_template: 系统提示模板
        model_type: 模型类型（chatglm, qwen 等）
        
    Returns:
        格式化后的对话字符串
    """
    system = item.get("system", system_template)
    conversations = item.get("conversation", [])
    
    # 根据模型类型选择格式化方式
    if model_type and "chatglm" in model_type.lower():
        return format_chatglm_conversation(item, system_template)
    elif model_type and "qwen" in model_type.lower():
        return format_qwen_conversation(item, system_template)
    else:
        # 默认通用格式
        formatted_text = f"<system>\n{system}\n</system>\n\n"
        
        for conv in conversations:
            human = conv.get("human", "")
            assistant = conv.get("assistant", "")
            
            formatted_text += f"<human>\n{human}\n</human>\n\n"
            formatted_text += f"<assistant>\n{assistant}\n</assistant>\n\n"
        
        return formatted_text.strip()


def format_chatglm_conversation(item: Dict, system_template: str = "你是一个名为沐雪的可爱AI女孩子") -> str:
    """
    格式化 ChatGLM 格式的对话
    
    Args:
        item: 包含 system 和 conversation 的字典
        system_template: 系统提示模板
        
    Returns:
        格式化后的对话字符串
    """
    system = item.get("system", system_template)
    conversations = item.get("conversation", [])
    
    # ChatGLM 格式：[Round 1]\n\n问：...\n\n答：...
    formatted_text = f"[Round 0]\n\n问：系统提示：{system}\n\n答：好的，我明白了。\n\n"
    
    round_num = 1
    for conv in conversations:
        human = conv.get("human", "")
        assistant = conv.get("assistant", "")
        
        formatted_text += f"[Round {round_num}]\n\n问：{human}\n\n答：{assistant}\n\n"
        round_num += 1
    
    return formatted_text.strip()


def format_qwen_conversation(item: Dict, system_template: str = "你是一个名为沐雪的可爱AI女孩子") -> str:
    """
    格式化 Qwen 格式的对话
    
    Args:
        item: 包含 system 和 conversation 的字典
        system_template: 系统提示模板
        
    Returns:
        格式化后的对话字符串
    """
    system = item.get("system", system_template)
    conversations = item.get("conversation", [])
    
    # Qwen 格式使用特殊 token
    formatted_text = f"<|im_start|>system\n{system}<|im_end|>\n"
    
    for conv in conversations:
        human = conv.get("human", "")
        assistant = conv.get("assistant", "")
        
        formatted_text += f"<|im_start|>user\n{human}<|im_end|>\n"
        formatted_text += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    
    return formatted_text

