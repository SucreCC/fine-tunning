"""
训练回调函数模块
logging / generation callback
"""
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Optional

from core.utils import logging

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.get_logger(__name__)

class GenerationCallback(TrainerCallback):
    """生成回调，用于在训练过程中生成示例文本"""
    
    def __init__(self, tokenizer, prompts: Optional[list] = None, generation_config: Optional[dict] = None):
        """
        初始化生成回调
        
        Args:
            tokenizer: 分词器
            prompts: 提示词列表
            generation_config: 生成配置
        """
        self.tokenizer = tokenizer
        self.prompts = prompts or ["你好", "介绍一下你自己"]
        self.generation_config = generation_config or {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
        }
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, model=None, logs=None, **kwargs):
        """在日志记录时生成示例文本"""
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            model.eval()
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, **self.generation_config)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                logger.info(f"\n[Step {state.global_step}] Prompt: {prompt}")
                logger.info(f"Generated: {generated_text}\n")
                
                # 如果使用 wandb，记录生成结果
                if wandb and wandb.run is not None:
                    wandb.log({
                        f"generation/{prompt[:20]}": generated_text,
                    }, step=state.global_step)
            
            model.train()


class WandbCallback(TrainerCallback):
    """Wandb 回调，用于初始化 wandb"""
    
    def __init__(self, wandb_config):
        """
        初始化 Wandb 回调
        
        Args:
            wandb_config: Wandb 配置
        """
        self.wandb_config = wandb_config
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时初始化 wandb"""
        if self.wandb_config.use_wandb:
            try:
                if self.wandb_config.wandb_api_key:
                    wandb.login(key=self.wandb_config.wandb_api_key)
                
                wandb.init(
                    project=self.wandb_config.wandb_project,
                    name=self.wandb_config.wandb_run_name,
                    entity=self.wandb_config.wandb_entity,
                    tags=self.wandb_config.wandb_tags,
                    dir=self.wandb_config.wandb_dir,
                )
            except ImportError:
                logger.info("警告: wandb 未安装，跳过 wandb 初始化")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时结束 wandb"""
        if self.wandb_config.use_wandb and wandb and wandb.run is not None:
            wandb.finish()



