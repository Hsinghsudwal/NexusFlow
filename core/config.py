import os
import yaml
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class to handle loading, saving, and accessing configuration values."""

    def __init__(self, config_path: str = None):
        """Initialize configuration manager and optionally load configuration.

        Args:
            config_path (str): Path to YAML config file.
        """
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML config file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value for a specific key.

        Args:
            key (str): The key of the configuration item.
            default (Any): The default value if the key is not found.

        Returns:
            Any: The configuration value for the given key.
        """
        value = self.config.get(key, default)
        logger.debug(f"Config: {key} = {value}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key (str): Configuration key.
            value (Any): Value to set.
        """
        self.config[key] = value
        logger.info(f"Config set: {key} = {value}")

    def save(self, path: str) -> None:
        """Save the current configuration to a YAML file.

        Args:
            path (str): The path to save the configuration file.
        """
        with open(path, "w") as f:
            yaml.dump(self.config, f)
        logger.info(f"Configuration saved to {path}")



# from config import Config

# Initialize configuration manager
# config_manager = Config(config_path="config.yaml")

# Get configuration value
# pipeline_name = config_manager.get("pipeline_name", default="default_pipeline")
# print(f"Pipeline Name: {pipeline_name}")

# Set and save new configuration value
# config_manager.set("model_version", "v1.0")
# config_manager.save("config.yaml")
