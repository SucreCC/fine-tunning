"""
推理 / 对比评估脚本
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.config import load_config
from core.model import load_model_and_tokenizer
from core.eval import generate_text, evaluate_generation


def main():
    """主评估函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="模型路径（如果不提供则使用配置文件中的路径）"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["你好", "介绍一下你自己"],
        help="测试提示词列表"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print("加载配置文件...")
    config = load_config(args.config)
    
    # 确定模型路径
    model_path = args.model_path or config.model_config.output_dir
    
    # 加载模型和分词器
    print(f"加载模型: {model_path}")
    # 这里需要根据实际情况调整模型加载逻辑
    # 如果是 LoRA 模型，需要使用 PeftModel.from_pretrained
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 尝试加载 LoRA 模型
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_config.base_model_path,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print("加载 LoRA 模型成功")
    except:
        # 如果不是 LoRA 模型，加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        print("加载完整模型成功")
    
    model.eval()
    
    # 评估生成
    print("\n开始评估生成...")
    results = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        test_prompts=args.prompts,
        max_new_tokens=100,
        temperature=0.7,
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print("-" * 50)


if __name__ == "__main__":
    main()

