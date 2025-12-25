"""
工具模块
"""
from .seed import set_seed

# 不在 __init__.py 中导入 config，避免循环导入
# 因为 config_manager 会导入 file_utils，如果 __init__.py 导入 config，
# 而 config 又导入 config_manager，会导致循环导入
# 用户应该直接从 core.utils.config 导入 ConfigManager 和 load_config
__all__ = ["set_seed"]

