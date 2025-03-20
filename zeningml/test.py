from typing import Dict, Any, List, Optional
import os
import json
import yaml
import pickle
import logging
import inspect
import pandas as pd
from abc import ABC, abstractmethod

# -----------------------------
# Config
# -----------------------------

class Config:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
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

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._artifacts: Dict[str, Any] = {}
        self.base_path = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> str:
        """Save an artifact and return its path."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        if name.endswith(".pkl"):
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
        elif name.endswith(".csv"):
            if isinstance(artifact, pd.DataFrame):
                artifact.to_csv(artifact_path, index=False)
            else:
                raise ValueError("CSV format only supports pandas DataFrames.")
        else:
            raise ValueError(f"Unsupported format for {name}")

        return artifact_path

    def load_artifact(self, subdir: str, name: str) -> Optional[Any]:
        """Load an artifact from storage."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    return pickle.load(f)
            elif name.endswith(".csv"):
                return pd.read_csv(artifact_path)
            else:
                raise ValueError(f"Unsupported format for {name}")
        return None

    def store(self, key: str, data: Any) -> str:
        """Store an artifact in memory and return its key."""
        self._artifacts[key] = data
        return key
    
    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve an artifact by key."""
        return self._artifacts.get(key)

# -----------------------------
# Step
# -----------------------------

class Step:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def execute(self, context: Dict[str, Any], artifact_store: ArtifactStore):
        sig = inspect.signature(self.func)
        kwargs = {param.name: artifact_store.get_artifact(context[param.name]) for param in sig.parameters.values() if param.name in context}
        return self.func(**kwargs)

# -----------------------------
# Pipeline
# -----------------------------

class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps: List[Step] = []

    def step(self, func):
        step_instance = Step(func)
        self.steps.append(step_instance)
        return step_instance

    def run(self, stack: "Stack"):
        context = {}
        artifact_store = stack.artifact_store

        for step in self.steps:
            output = step.execute(context, artifact_store)
            artifact_id = artifact_store.store(step.name, output)
            context[step.name] = artifact_id

# -----------------------------
# Decorators
# -----------------------------

def step(func):
    return Step(func)

def pipeline(name: str):
    def decorator(func):
        pipe = Pipeline(name)
        func(pipe)
        return pipe
    return decorator

# -----------------------------
# Stack
# -----------------------------

class Stack:
    def __init__(self, name: str, artifact_store: ArtifactStore):
        self.name = name
        self.artifact_store = artifact_store

class S3ArtifactStore(Stack):
    def store_artifact(self, artifact: Any, artifact_name: str) -> str:
        """Implement S3 storage"""
        return f"s3://bucket/{artifact_name}"

class Orchestrator(Stack, ABC):
    @abstractmethod
    def execute_pipeline(self, pipeline: Pipeline, stack: Stack):
        """Execute the pipeline"""
        pass

class KubeflowOrchestrator(Orchestrator):
    def execute_pipeline(self, pipeline: Pipeline, stack: Stack):
        """Logic for executing pipeline with Kubeflow"""
        pass

class MetadataStore(Stack):
    def log_metadata(self, metadata: Dict[str, Any], step_name: str):
        """Log metadata for a pipeline step"""
        pass

class SQLiteMetadataStore(Stack):
    def __init__(self, db_path: str = "metadata.db"):
        super().__init__("sqlite_metadata_store")
        import sqlite3  # Lazy import
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS metadata
                             (id INTEGER PRIMARY KEY,
                              step_name TEXT,
                              metadata TEXT)''')
    
    def log_metadata(self, metadata: Dict, step_name: str):
        self.conn.execute("INSERT INTO metadata (step_name, metadata) VALUES (?, ?)",
                          (step_name, json.dumps(metadata)))
        self.conn.commit()

# -----------------------------
# Example
# -----------------------------

if __name__ == "__main__":
    
    @pipeline("example_pipeline")
    def my_pipeline(pipe: Pipeline):
        @pipe.step
        def first_step():
            return "Initial data"

        @pipe.step
        def second_step(first_step: str):
            return f"Processed: {first_step}"

    # Create stack with artifact store
    artifact_store = ArtifactStore()

    
    s3_store = S3ArtifactStore(name="s3", artifact_store=artifact_store)
    dev_stack = Stack(name="dev", artifact_store=artifact_store)

    # Run pipeline
    pipeline_instance = my_pipeline  # Get pipeline instance
    pipeline_instance.run(dev_stack)

    # Verify artifacts
    print(artifact_store._artifacts)  # Shows stored artifacts
