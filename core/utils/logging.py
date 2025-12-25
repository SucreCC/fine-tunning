"""
日志配置模块

提供统一的日志配置和日志记录器。
支持控制台和文件输出，可配置日志级别和格式。
"""
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from core.dto.config.log_config import LogConfig


def setup_logging(config: LogConfig) -> None:
    """
    配置日志系统

    设置根日志记录器，配置控制台和文件处理器。
    
    Args:
        config: 日志配置对象
    """
    
    # 保证日志目录的存在
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 将字符串日志级别转换为 logging 级别
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有的处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(config.log_format, config.date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器（所有日志）
    file_handler = TimedRotatingFileHandler(
        config.log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(config.log_format, config.date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 错误日志文件处理器（只记录 ERROR 及以上级别）
    error_file_handler = TimedRotatingFileHandler(
        config.error_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_file_handler)

    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    root_logger.info("日志系统初始化完成")


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
    
    Returns:
        logging.Logger: 配置好的日志记录器
    
    Example:
        logger = get_logger(__name__)
        logger.info("这是一条信息日志")
    """
    return logging.getLogger(name or __name__)
