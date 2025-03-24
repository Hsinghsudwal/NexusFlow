import json
import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")


import os
import pickle
import pandas as pd
import logging
from typing import Any

class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config):
        self.config = config
        self.base_path = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> str:
        """Save an artifact in the specified format and return the path."""
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
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(self, subdir: str, name: str):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
            else:
                raise ValueError(f"Unsupported format for {name}")
            logging.info(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
        else:
            logging.warning(f"Artifact '{name}' not found in {artifact_path}")
            return None

    def list_artifacts(self, run_id=None):
        """List all artifacts or artifacts for a specific run."""
        artifacts = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts


from functools import wraps
from typing import List, Callable

def node(name: str = None, stage: int = None, dependencies: List[str] = None):
    """Decorator to mark a function as a step in a pipeline with rich metadata."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Executing step: {func.__name__}")
            result = func(*args, **kwargs)
            print(f"Completed step: {func.__name__} with result: {result}")
            return result

        wrapper._is_node = True  # Mark as a node for discovery
        wrapper._node_metadata = {
            "name": name or func.__name__,
            "stage": stage or 0,
            "dependencies": dependencies or []
        }
        return wrapper

    return decorator


import logging
import uuid
from typing import Dict, Any, Callable, Optional

class Stack:
    """A stack that brings together all components needed to run pipelines and manages their execution."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.artifact_store = None
        self.run_id = None
        self.pipeline_func = None
        self.pipeline_args = ()
        self.pipeline_kwargs = {}

    def set_artifact_store(self, artifact_store):
        """Set the artifact store for the stack."""
        self.artifact_store = artifact_store
        return self

    def set_pipeline(self, func: Callable, *args, **kwargs):
        """Set the pipeline function and its arguments."""
        self.pipeline_func = func
        self.pipeline_args = args
        self.pipeline_kwargs = kwargs
        return self

    def run(self, run_id=None):
        """Execute the pipeline with the given run_id."""
        self.run_id = run_id or str(uuid.uuid4())
        logging.info(f"Starting pipeline execution with run_id: {self.run_id}")

        if self.pipeline_func is None:
            raise ValueError("Pipeline function is not set.")

        print(f"Running pipeline: {self.pipeline_func.__name__}")
        self.pipeline_func(*self.pipeline_args, **self.pipeline_kwargs)

        logging.info(f"Completed pipeline execution with run_id: {self.run_id}")
        return self.run_id

    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Mock implementation to get run details."""
        # This should be replaced with actual logic to fetch run details
        return {
            "pipeline_name": self.pipeline_func.__name__,
            "status": "completed",
            "duration_seconds": 120,
            "tasks": ["data_ingestion", "data_processing"]
        }


import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple
from config import Config
from artifact_store import ArtifactStore

class DataIngestion:
    """Handle data ingestion operations."""

    def data_ingestion(self, path: str, config: Config, artifact_store: ArtifactStore) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into train and test sets."""
        raw_path = config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = config.get("filenames", {}).get("raw_test", "test_data.csv")

        df = pd.read_csv(path)
        test_size = config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        logging.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        artifact_store.save_artifact(train_data, subdir=raw_path, name=raw_train_filename)
        artifact_store.save_artifact(test_data, subdir=raw_path, name=raw_test_filename)

        logging.info("Data ingestion completed")
        return train_data, test_data


import pandas as pd
import logging
from typing import Tuple
from config import Config
from artifact_store import ArtifactStore

class DataProcessor:
    """Handle data processing operations."""

    def process_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, config: Config, artifact_store: ArtifactStore) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process the train and test data."""
        processed_path = config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = config.get("filenames", {}).get("processed_train", "processed_train.csv")
        processed_test_filename = config.get("filenames", {}).get("processed_test", "processed_test.csv")

        processed_train = train_data.copy()
        processed_test = test_data.copy()

        artifact_store.save_artifact(processed_train, subdir=processed_path, name=processed_train_filename)
        artifact_store.save_artifact(processed_test, subdir=processed_path, name=processed_test_filename)

        logging.info("Data processing completed")
        return processed_train, processed_test


import logging
import uuid
from config import Config
from artifact_store import ArtifactStore
from stack import Stack
from data_ingestion import DataIngestion
from data_processor import DataProcessor

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        self.config = Config.load_file(config_path)
        self.artifact_store = ArtifactStore(self.config)
        self.stack = Stack("Training Pipeline", self.config.config_dict)
        self.stack.set_artifact_store(self.artifact_store)

    def run(self):
        """Run the complete pipeline."""
        run_id = str(uuid.uuid4())
        logging.info(f"Starting pipeline run with ID: {run_id}")

        data_ingestion = DataIngestion()
        data_processor = DataProcessor()

        train_data, test_data = data_ingestion.data_ingestion(self.data_path, self.config, self.artifact_store)
        processed_train, processed_test = data_processor.process_data(train_data, test_data, self.config, self.artifact_store)

        artifacts = self.artifact_store.list_artifacts(run_id)
        print(f"Run ID: {run_id}")
        print("Artifacts:")
        for uri in artifacts:
            print(f"- {uri}")

        run_details = self.stack.get_run_details(run_id)
        print("\nRun Details:")
        print(f"Pipeline: {run_details.get('pipeline_name')}")
        print(f"Status: {run_details.get('status')}")
        print(f"Duration: {run_details.get('duration_seconds', 0):.2f} seconds")
        print(f"Tasks executed: {', '.join(run_details.get('tasks', []))}")

        if run_details.get("status") == "completed":
            print("Pipeline completed successfully")

        return run_id

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_path = "data.csv"
    config_path = "config/config.yml"
    pipeline = TrainingPipeline(data_path, config_path)
    results = pipeline.run()
    logging.info(f"Pipeline execution results: {results}")
