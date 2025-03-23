# config/config.py
import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataConfig:
    source_path: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    
@dataclass
class ModelConfig:
    model_type: str
    hyperparameters: Dict[str, Any]
    
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping: bool
    patience: int
    
@dataclass
class MLOpsConfig:
    experiment_name: str
    run_id: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    artifact_dir: str
    db_connection: str

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config from a YAML file or use default."""
        default_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')
        self.config_path = config_path or default_path
        self.config = self._load_config()
        
    def _load_config(self) -> MLOpsConfig:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            
        # Convert dictionary to dataclass objects
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        
        # Create main config
        return MLOpsConfig(
            experiment_name=config_dict['experiment_name'],
            run_id=config_dict['run_id'],
            data=data_config,
            model=model_config,
            training=training_config,
            artifact_dir=config_dict['artifact_dir'],
            db_connection=config_dict['db_connection']
        )
    
    def get_config(self) -> MLOpsConfig:
        """Return the loaded configuration."""
        return self.config
    
    def update_config(self, new_config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        # Deep update of nested dictionaries
        # This is a simplified implementation - in production, you'd want
        # to handle nested updates more robustly
        config_dict = self._config_to_dict(self.config)
        self._deep_update(config_dict, new_config_dict)
        self._save_config(config_dict)
        self.config = self._load_config()
        
    def _config_to_dict(self, config: MLOpsConfig) -> Dict[str, Any]:
        """Convert config dataclass to dictionary."""
        result = {
            'experiment_name': config.experiment_name,
            'run_id': config.run_id,
            'data': {
                'source_path': config.data.source_path,
                'train_ratio': config.data.train_ratio,
                'val_ratio': config.data.val_ratio,
                'test_ratio': config.data.test_ratio,
                'random_state': config.data.random_state,
            },
            'model': {
                'model_type': config.model.model_type,
                'hyperparameters': config.model.hyperparameters,
            },
            'training': {
                'epochs': config.training.epochs,
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'early_stopping': config.training.early_stopping,
                'patience': config.training.patience,
            },
            'artifact_dir': config.artifact_dir,
            'db_connection': config.db_connection,
        }
        return result
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
                
    def _save_config(self, config_dict: Dict[str, Any]) -> None:
        """Save updated configuration to YAML file."""
        with open(self.config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)

# Example settings.yaml content
# config/settings.yaml (to be created separately)
"""
experiment_name: default_experiment
run_id: run_1
data:
  source_path: ./data/raw
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_state: 42
model:
  model_type: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  early_stopping: true
  patience: 3
artifact_dir: ./artifacts
db_connection: sqlite:///db/mlops.db
"""
