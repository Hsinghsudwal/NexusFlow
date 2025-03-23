# Core folder structure for the MLOps framework
'''
├── mlops_framework/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Logging utilities
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base pipeline class
│   │   ├── data_pipeline.py         # Data processing pipeline
│   │   ├── feature_pipeline.py      # Feature engineering pipeline
│   │   ├── training_pipeline.py     # Model training pipeline
│   │   ├── evaluation_pipeline.py   # Model evaluation pipeline
│   │   └── deployment_pipeline.py   # Model deployment pipeline
│   ├── components/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base component class
│   │   ├── data/                    # Data components
│   │   ├── preprocessing/           # Preprocessing components
│   │   ├── feature/                 # Feature engineering components
│   │   ├── model/                   # Model training components
│   │   ├── evaluation/              # Evaluation components
│   │   └── deployment/              # Deployment components
│   ├── versioning/
│   │   ├── __init__.py
│   │   ├── artifact_store.py        # Artifact versioning and storage
│   │   ├── model_registry.py        # Model registry for versioning
│   │   └── lineage.py               # Lineage tracking
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── data_drift.py            # Data drift detection
│   │   ├── model_drift.py           # Model drift detection
│   │   ├── performance.py           # Performance monitoring
│   │   └── alerting.py              # Alerting system
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── service.py               # Model serving
│   │   ├── container.py             # Container management
│   │   └── endpoints.py             # API endpoints
│   └── utils/
│       ├── __init__.py
│       ├── io.py                    # I/O utilities
│       ├── validation.py            # Data validation utilities
│       └── metrics.py               # Metrics calculation utilities
├── examples/                        # Example pipelines and workflows
├── tests/                           # Unit and integration tests
├── setup.py                         # Package setup file
└── README.md                        # Documentation
'''

# config.py - Configuration management
import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class MLOpsConfig:
    """Configuration for MLOps framework."""
    project_name: str
    artifact_store_path: str
    model_registry_path: str
    experiment_tracking_uri: Optional[str] = None
    deployment_target: str = "local"
    monitoring_config: Dict[str, Any] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MLOpsConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

# logger.py - Logging utilities
import logging
import sys
from datetime import datetime

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Set up logger with specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# pipeline/base.py - Base pipeline class
import uuid
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import datetime
import json
import os

class Pipeline(ABC):
    """Base class for all pipelines in the MLOps framework."""
    
    def __init__(self, name: str, config: MLOpsConfig):
        self.name = name
        self.config = config
        self.id = str(uuid.uuid4())
        self.components = []
        self.artifacts = {}
        self.metadata = {
            "pipeline_id": self.id,
            "pipeline_name": self.name,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "initialized",
            "version": "1.0.0",
        }
        self.logger = setup_logger(f"pipeline.{self.name}")
    
    def add_component(self, component):
        """Add a component to the pipeline."""
        self.components.append(component)
        self.logger.info(f"Added component: {component.name}")
        return self
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the pipeline."""
        pass
    
    def save_metadata(self):
        """Save pipeline metadata to artifact store."""
        metadata_path = os.path.join(
            self.config.artifact_store_path, 
            self.name, 
            self.id, 
            "metadata.json"
        )
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path
    
    def save_artifact(self, name: str, artifact: Any, artifact_type: str):
        """Save an artifact to the artifact store."""
        from ..versioning.artifact_store import save_artifact
        
        artifact_path = save_artifact(
            artifact=artifact,
            name=name,
            artifact_type=artifact_type,
            pipeline_id=self.id,
            pipeline_name=self.name,
            config=self.config
        )
        
        self.artifacts[name] = {
            "path": artifact_path,
            "type": artifact_type,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Saved artifact {name} to {artifact_path}")
        return artifact_path

# components/base.py - Base component class
class Component(ABC):
    """Base class for all components in the MLOps framework."""
    
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())
        self.metadata = {
            "component_id": self.id,
            "component_name": self.name,
            "created_at": datetime.datetime.now().isoformat(),
        }
        self.logger = setup_logger(f"component.{self.name}")
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the component logic."""
        pass
    
    def log_metadata(self, key: str, value: Any):
        """Log metadata for the component."""
        self.metadata[key] = value
        self.logger.info(f"Logged metadata: {key}={value}")

# versioning/artifact_store.py - Artifact versioning and storage
import os
import pickle
import json
import shutil
from typing import Any, Dict, Optional

def save_artifact(
    artifact: Any,
    name: str,
    artifact_type: str,
    pipeline_id: str,
    pipeline_name: str,
    config: MLOpsConfig
) -> str:
    """Save an artifact to the artifact store with versioning."""
    # Create artifact path
    artifact_dir = os.path.join(
        config.artifact_store_path,
        pipeline_name,
        pipeline_id,
        "artifacts",
        name
    )
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Determine file extension based on artifact type
    extensions = {
        "model": ".pkl",
        "data": ".parquet",
        "metrics": ".json",
        "plot": ".png",
        "feature": ".json",
        "config": ".yaml",
    }
    extension = extensions.get(artifact_type, ".pkl")
    
    # Full path to artifact
    artifact_path = os.path.join(artifact_dir, f"{name}{extension}")
    
    # Save artifact based on type
    if artifact_type == "model":
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
    elif artifact_type == "metrics" or artifact_type == "feature":
        with open(artifact_path, 'w') as f:
            json.dump(artifact, f, indent=2)
    elif artifact_type == "data":
        if hasattr(artifact, 'to_parquet'):
            artifact.to_parquet(artifact_path)
        else:
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
    else:
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
    
    # Save metadata
    metadata = {
        "name": name,
        "type": artifact_type,
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_name,
        "created_at": datetime.datetime.now().isoformat(),
        "path": artifact_path
    }
    
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return artifact_path

def load_artifact(
    name: str,
    pipeline_id: str,
    pipeline_name: str,
    config: MLOpsConfig
) -> Any:
    """Load an artifact from the artifact store."""
    # Create artifact path
    artifact_dir = os.path.join(
        config.artifact_store_path,
        pipeline_name,
        pipeline_id,
        "artifacts",
        name
    )
    
    # Load metadata to determine file extension
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    artifact_path = metadata["path"]
    artifact_type = metadata["type"]
    
    # Load artifact based on type
    if artifact_type == "model":
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)
    elif artifact_type == "metrics" or artifact_type == "feature":
        with open(artifact_path, 'r') as f:
            return json.load(f)
    elif artifact_type == "data":
        if artifact_path.endswith('.parquet'):
            import pandas as pd
            return pd.read_parquet(artifact_path)
        else:
            with open(artifact_path, 'rb') as f:
                return pickle.load(f)
    else:
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)

# versioning/model_registry.py - Model registry
class ModelRegistry:
    """Model registry for versioning and managing models."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.registry_path = config.model_registry_path
        os.makedirs(self.registry_path, exist_ok=True)
        self.logger = setup_logger("model_registry")
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        version: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Register a model in the registry."""
        # Create model directory
        model_dir = os.path.join(self.registry_path, model_name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model to registry
        registry_model_path = os.path.join(model_dir, f"{model_name}.pkl")
        shutil.copy2(model_path, registry_model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Registered model {model_name} version {version}")
        return registry_model_path
    
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a model from the registry."""
        if version == "latest":
            # Find latest version
            model_dir = os.path.join(self.registry_path, model_name)
            versions = os.listdir(model_dir)
            versions.sort(reverse=True)
            version = versions[0]
        
        model_path = os.path.join(
            self.registry_path,
            model_name,
            version,
            f"{model_name}.pkl"
        )
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.logger.info(f"Loaded model {model_name} version {version}")
        return model
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """Get model metadata from the registry."""
        if version == "latest":
            # Find latest version
            model_dir = os.path.join(self.registry_path, model_name)
            versions = os.listdir(model_dir)
            versions.sort(reverse=True)
            version = versions[0]
        
        metadata_path = os.path.join(
            self.registry_path,
            model_name,
            version,
            "metadata.json"
        )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self) -> List[str]:
        """List all models in the registry."""
        return os.listdir(self.registry_path)
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        model_dir = os.path.join(self.registry_path, model_name)
        versions = os.listdir(model_dir)
        versions.sort(reverse=True)
        return versions
    
    def promote_model_to_production(self, model_name: str, version: str) -> str:
        """Promote a model version to production."""
        # Create production directory
        production_dir = os.path.join(self.registry_path, "production")
        os.makedirs(production_dir, exist_ok=True)
        
        # Create symlink to the model version
        source_path = os.path.join(self.registry_path, model_name, version)
        target_path = os.path.join(production_dir, model_name)
        
        # Remove existing symlink if it exists
        if os.path.exists(target_path):
            if os.path.islink(target_path):
                os.unlink(target_path)
            else:
                shutil.rmtree(target_path)
        
        # Create symlink
        os.symlink(source_path, target_path)
        
        # Update production models list
        production_models_path = os.path.join(production_dir, "production_models.json")
        production_models = {}
        
        if os.path.exists(production_models_path):
            with open(production_models_path, 'r') as f:
                production_models = json.load(f)
        
        production_models[model_name] = {
            "version": version,
            "promoted_at": datetime.datetime.now().isoformat()
        }
        
        with open(production_models_path, 'w') as f:
            json.dump(production_models, f, indent=2)
        
        self.logger.info(f"Promoted model {model_name} version {version} to production")
        return target_path
