import os
from pathlib import Path


def find_project_root(config_file_name: str = "config.yaml") -> Path:
    """
    查找项目根路径（config.yaml 所在的文件夹）

    从当前工作目录一级一级往上找，直到找到指定的配置文件

    Args:
        config_file_name: 配置文件名，默认为 "config.yaml"

    Returns:
        项目根目录的绝对路径

    Raises:
        FileNotFoundError: 如果找不到配置文件
    """
    # 从当前工作目录开始向上查找
    current_path = Path(os.getcwd()).absolute()

    # 向上查找，直到找到包含配置文件的目录
    while current_path != current_path.parent:
        config_file = current_path / config_file_name
        if config_file.exists():
            return current_path
        current_path = current_path.parent

    # 如果找不到，抛出异常
    raise FileNotFoundError(
        f"找不到配置文件 '{config_file_name}'。请确保在项目根目录或其子目录中运行。"
    )