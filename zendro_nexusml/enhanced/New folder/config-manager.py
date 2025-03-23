# mlops_framework/config.py

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Singleton class to manage configuration for the MLOps framework.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration."""
        self.config_path = os.environ.get("CONFIG_PATH", "config.yaml")
        self.config = self._load_config()
        logger.info(f"Initialized config manager with config path {self.config_path}")
    
    @lru_cache(maxsize=1)
    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "metadata": {
                "db_path": "metadata.db"
            },
            "artifacts": {
                "store_path": "artifacts",
                "backend": "local"
            },
            "cache": {
                "enabled": True
            },
            "integrations": {
                "mlflow": {
                    "enabled": False,
                    "tracking_uri": ""
                },
                "cloud": {
                    "provider": "",
                    "credentials": {}
                }
            }
        }
        
        # Try to load config from file
        if os.path.exists(self.config_path):
            try:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                    with open(self.config_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                elif self.config_path.endswith(".json"):
                    with open(self.config_path, 'r') as f:
                        file_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}")
                    file_config = {}
                
                # Merge default with file config
                self._deep_update(default_config, file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {self.config_path}: {str(e)}")
        else:
            logger.warning(f"Config file {self.config_path} not found, using default configuration")
        
        # Apply environment variable overrides
        self._apply_env_overrides(default_config)
        
        return default_config
    
    def _deep_update(self, d: Dict, u: Dict):
        """
        Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Update values
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def _apply_env_overrides(self, config: Dict):
        """
        Apply environment variable overrides to config.
        
        Args:
            config: Configuration dictionary to update
        """
        # Mapping from env var to config path
        env_mappings = {
            "MLOPS_LOG_LEVEL": ["logging", "level"],
            "MLOPS_METADATA_DB_PATH": ["metadata", "db_path"],
            "MLOPS_ARTIFACT_STORE_PATH": ["artifacts", "store_path"],
            "MLOPS_ARTIFACT_BACKEND": ["artifacts", "backend"],
            "MLOPS_CACHE_ENABLED": ["cache", "enabled"],
            "MLOPS_MLFLOW_ENABLED": ["integrations", "mlflow", "enabled"],
            "MLOPS_MLFLOW_TRACKING_URI": ["integrations", "mlflow", "tracking_uri"],
            "MLOPS_CLOUD_PROVIDER": ["integrations", "cloud", "provider"]
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                # Navigate to the right spot in the config
                current = config
                for i, key in enumerate(config_path):
                    if i == len(config_path) - 1:
                        # Set the value with the right type
                        value = os.environ[env_var]
                        if isinstance(current[key], bool):
                            current[key] = value.lower() in ("true", "yes", "1")
                        elif isinstance(current[key], int):
                            current[key] = int(value)
                        elif isinstance(current[key], float):
                            current[key] = float(value)
                        else:
                            current[key] = value
                    else:
                        current = current[key]
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.
        
        Args:
            path: Dot-separated path to the config value
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value
        """
        keys = path.split("