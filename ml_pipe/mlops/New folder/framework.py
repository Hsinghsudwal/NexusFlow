import os
import uuid
import yaml
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional


# -----------------------------
# Config
# -----------------------------

class Config:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with optional default."""
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str) -> "Config":
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")





# -----------------------------
# Artifact Store
# -----------------------------

class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config: Optional[Config] = None, run_id: str = None):
        self.config = config or Config()
        self.run_id = run_id or "latest"
        self._artifacts: Dict[str, Any] = {}
        
        # Set up base path with run_id for versioning
        base_dir = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        self.base_path = os.path.join(base_dir, self.run_id)
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> str:
        """Save an artifact to disk and return its path."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        if name.endswith(".pkl"):
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
        else:
            raise ValueError(f"Unsupported format for {name}")
            
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(self, subdir: str, name: str) -> Optional[Any]:
        """Load an artifact from disk."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format for {name}")
            logging.info(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
        else:
            logging.warning(f"Artifact '{name}' not found at {artifact_path}")
            return None

    def store(self, key: str, data: Any) -> str:
        """Store an artifact in memory and return its key."""
        self._artifacts[key] = data
        return key
    
    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve an in-memory artifact by key."""
        return self._artifacts.get(key)
        
    def list_artifacts(self, run_id: Optional[str] = None) -> List[str]:
        """List all artifacts for a given run."""
        target_path = os.path.join(
            self.config.get("folder_path", {}).get("artifacts", "artifacts"),
            run_id or self.run_id
        )
        
        if not os.path.exists(target_path):
            return []
            
        artifacts = []
        for root, _, files in os.walk(target_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), target_path)
                artifacts.append(rel_path)
        
        return artifacts



# -----------------------------
# Stack
# -----------------------------

class Stack:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = Config(config or {})

# ----------------------------- 
# Example Usage
# ----------------------------- 
class DataIngestion:

    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def data_ingestion(self, path):
        # Load raw data
        df = pd.read_csv(path)
        # Split data
        test_size = self.config.get("base",{}).get("test_size",{})
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        self.artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename 
        )
        self.artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename
        )
        return train_data, test_data


# ----------------------------- 
# Pipe
# -----------------------------
class TrainingPipeline:


    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml").config_dict
        self.stack.dataingest = DataIngestion(self.config)


    def run_example():

        # Create stack
        dev_stack = Stack(name="simple_pipeline", config=config)

        train, test = self.stack.dataingest.data_ingestion(self.path)

        # Create and run pipeline
        pipeline = Pipeline(name="example_pipeline", description="A simple example pipeline")
  
        run_id = pipeline.run(dev_stack, tags={"env": "dev"})

        # List artifacts
        artifacts = dev_stack.artifact_store.list_artifacts(run_id)
        print(f"Run ID: {run_id}")
        print("Artifacts:")
        for uri in artifacts:
            print(f"- {uri}")

        # Get run details
        run_details = ml_stack.get_run_details(run_id)
        print("\nRun Details:")
        print(f"Pipeline: {run_details['pipeline_name']}")
        print(f"Status: {run_details['status']}")

        # Register the model in the registry (if run was successful)
        if run_details["status"] == "completed":
            print("Pipeline successfully")

        return run_id, dev_stack


if __name__ == "__main__":
    path = "data.csv"
    pipeline = TrainingPipeline(path)
    pipeline.run_example()
