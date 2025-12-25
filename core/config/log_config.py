from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LogConfig:
    service_name: str = "unknown service"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_dir: str = "./logs"
    log_file: str = ""
    error_file: str = "aiabo.error"
    log_level: str = "INFO"
    uvicorn_log_level: str = "INFO"
    uvicorn_access_log_level: str = "WARNING"
    sqlalchemy_log_level: str = "WARNING"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LogConfig':
        """从字典创建配置对象"""
        return cls(
            service_name=config_dict.get('service_name', ""),
            log_format=config_dict.get('log_format', ""),
            date_format=config_dict.get('date_format', ""),
            log_dir=config_dict.get('log_dir', ""),
            log_file=config_dict.get('log_file', ""),
            error_file=config_dict.get('error_file', ""),
            log_level=config_dict.get('log_level', ""),
            uvicorn_log_level=config_dict.get('uvicorn_log_level', ""),
            uvicorn_access_log_level=config_dict.get('uvicorn_access_log_level', ""),
            sqlalchemy_log_level=config_dict.get('sqlalchemy_log_level', ""),

        )
