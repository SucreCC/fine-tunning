from core.config.config_manager import ConfigManager

if __name__ == "__main__":
    config_manager = ConfigManager.from_yaml()
    print(config_manager)