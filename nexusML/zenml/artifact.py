import os
import json
import pickle
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Tuple
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the MLOps framework"""
    
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_config(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return ConfigManager(config_data)  # Changed to return instance of ConfigManager
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")

    @staticmethod
    def save_config(config: Dict, config_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
            logger.info(f"Config saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise


class ArtifactStore:
    """Store for managing pipeline artifacts"""
    
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, name: str, obj):
        """Save an object as an artifact"""
        path = os.path.join(self.base_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved artifact: {path}")
    
    def load(self, name: str):
        """Load an object from an artifact"""
        path = os.path.join(self.base_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def save_artifact(self, artifact: Any, name: str, metadata: Dict = None) -> str:
        """Save an artifact to the store with metadata"""
        try:
            # Create unique artifact ID
            artifact_id = str(uuid.uuid4())
            artifact_dir = os.path.join(self.base_dir, artifact_id)
            os.makedirs(artifact_dir, exist_ok=True)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "name": name,
                "created_at": datetime.now().isoformat(),
                "artifact_id": artifact_id
            })
            
            # Save metadata as JSON
            with open(os.path.join(artifact_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f)
            
            # Save the artifact based on its type
            artifact_path = os.path.join(artifact_dir, "artifact.pkl")
            
            if isinstance(artifact, pd.DataFrame):
                artifact.to_pickle(artifact_path)
            else:
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact, f)
            
            logger.info(f"Artifact {name} saved with ID {artifact_id}")
            return artifact_id
            
        except Exception as e:
            logger.error(f"Error saving artifact {name}: {e}")
            raise
    
    def load_artifact(self, artifact_id: str) -> Tuple[Any, Dict]:
        """Load an artifact and its metadata from the store"""
        try:
            artifact_dir = os.path.join(self.base_dir, artifact_id)
            
            # Load metadata
            with open(os.path.join(artifact_dir, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            # Load the artifact
            artifact_path = os.path.join(artifact_dir, "artifact.pkl")
            
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
            
            with open(artifact_path, 'rb') as f:
                artifact = pickle.load(f)
            
            logger.info(f"Artifact {metadata['name']} loaded from ID {artifact_id}")
            return artifact, metadata
            
        except Exception as e:
            logger.error(f"Error loading artifact {artifact_id}: {e}")
            raise
    
    def list_artifacts(self, filter_func: Callable = None) -> List[Dict]:
        """List artifacts in the store, optionally filtered"""
        try:
            artifacts = []
            
            for artifact_id in os.listdir(self.base_dir):
                artifact_dir = os.path.join(self.base_dir, artifact_id)
                if os.path.isdir(artifact_dir):
                    metadata_path = os.path.join(artifact_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        if filter_func is None or filter_func(metadata):
                            artifacts.append(metadata)
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error listing artifacts: {e}")
            raise
