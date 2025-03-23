# config/config.py
import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class ModelConfig:
    name: str
    version: str
    framework: str
    params: Dict[str, Any]


@dataclass
class PipelineConfig:
    name: str
    version: str
    steps: List[str]
    artifacts_dir: str


@dataclass
class DeploymentConfig:
    target: str  # 'local', 'docker', 'kubernetes', etc.
    endpoint_name: str
    resources: Dict[str, Any]
    monitoring_enabled: bool


@dataclass
class MLOpsConfig:
    model: ModelConfig
    pipeline: PipelineConfig
    deployment: DeploymentConfig
    experiment_tracking: Dict[str, Any]
    local_db: Dict[str, Any]
    logging: Dict[str, Any]


class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> MLOpsConfig:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        model_config = ModelConfig(**config_dict.get('model', {}))
        pipeline_config = PipelineConfig(**config_dict.get('pipeline', {}))
        deployment_config = DeploymentConfig(**config_dict.get('deployment', {}))
        
        return MLOpsConfig(
            model=model_config,
            pipeline=pipeline_config,
            deployment=deployment_config,
            experiment_tracking=config_dict.get('experiment_tracking', {}),
            local_db=config_dict.get('local_db', {}),
            logging=config_dict.get('logging', {})
        )
    
    def get_config(self) -> MLOpsConfig:
        """Get the loaded configuration."""
        return self.config
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a specific configuration value."""
        if not hasattr(self.config, section):
            raise ValueError(f"Config section not found: {section}")
            
        section_config = getattr(self.config, section)
        if not hasattr(section_config, key):
            raise ValueError(f"Config key not found in section {section}: {key}")
            
        setattr(section_config, key, value)
    
    def save_config(self) -> None:
        """Save the current configuration back to YAML file."""
        # Convert the dataclass to a dictionary
        config_dict = {
            'model': vars(self.config.model),
            'pipeline': vars(self.config.pipeline),
            'deployment': vars(self.config.deployment),
            'experiment_tracking': self.config.experiment_tracking,
            'local_db': self.config.local_db,
            'logging': self.config.logging
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(f"Model name: {config.model.name}")
    print(f"Pipeline version: {config.pipeline.version}")
