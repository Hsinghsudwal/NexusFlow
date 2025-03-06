# Configuration Management (core/config.py)
import yaml
from typing import Dict, Any
import os

class ConfigManager:
    def __init__(self, config_path=None):
        """
        Manage configuration for NexusML pipelines
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config: Dict[str, Any] = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from YAML file
        
        Args:
            config_path (str): Path to configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
    
    def get(self, key, default=None):
        """
        Get configuration value
        
        Args:
            key (str): Configuration key
            default (Any, optional): Default value if key not found
        
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def update(self, key, value):
        """
        Update configuration value
        
        Args:
            key (str): Configuration key
            value (Any): Configuration value
        """
        self.config[key] = value
    
    def save_config(self, config_path):
        """
        Save current configuration to YAML file
        
        Args:
            config_path (str): Path to save configuration file
        """
        with open(config_path, 'w') as config_file:
            yaml.dump(self.config, config_file)