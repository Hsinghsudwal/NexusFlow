import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                # If run_id is specified, only include artifacts containing that run_id
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts

# Decorator to create task functions
def node(func=None, name: str = None, stage: int = None):
    """Decorator to mark functions as pipeline node."""
    def decorator(func):
        # We'll keep the original function but add metadata
        func._stage_metadata = {
            "name": name or func.__name__,
            "stage": stage
        }
        return func
    return decorator

class Pipeline:
    def __init__(self):
        self.tasks = []

    def add_task(self, task_func, *args, **kwargs):
        self.tasks.append((task_func, args, kwargs))

    def run(self):
        for task_func, args, kwargs in self.tasks:
            task_func(*args, **kwargs)


class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    @node(name="data_loader", task=1)
    def data_ingestion(self, path):
        # Define paths for artifacts
        raw_path = self.config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = self.config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = self.config.get("filenames", {}).get("raw_test", "test_data.csv")

        # Load raw data
        df = pd.read_csv(path)
        # Split data
        test_size = self.config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        # Save raw artifacts
        self.artifact_store.save_artifact(train_data, subdir=raw_path, name=raw_train_filename)
        self.artifact_store.save_artifact(test_data, subdir=raw_path, name=raw_test_filename)
        logging.info("Data ingestion completed")
        return train_data, test_data

class DataProcess:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    @node(name="data_process", task=2)
    def data_processing(self, train_data, test_data):
        # Define paths for artifacts
        process_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        process_train_filename = self.config.get("filenames", {}).get("processed_train", "processed_train_data.csv")
        process_test_filename = self.config.get("filenames", {}).get("processed_test", "processed_test_data.csv")

        # Process logic (dummy processing for illustration)
        train_process = train_data.copy()  # Replace with actual processing logic
        test_process = test_data.copy()  # Replace with actual processing logic

        # Save artifacts
        self.artifact_store.save_artifact(train_process, subdir=process_path, name=process_train_filename)
        self.artifact_store.save_artifact(test_process, subdir=process_path, name=process_test_filename)
        logging.info("Data process completed")
        return train_process, test_process

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""
    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        self.config = config_path

    def run(self):
        dataingest = DataIngestion()
        dataprocess = DataProcess()

        pipeline = Pipeline()
        pipeline.add_task(dataingest.data_ingestion, self.data_path, self.config)
        pipeline.add_task(dataprocess.data_processing, dataingest.data_ingestion(self.data_path)[0], dataingest.data_ingestion(self.data_path)[1], self.config)
        pipeline.run()

if __name__ == "__main__":
    data_path = "data.csv"
    config_path = "config/config.yml"

    pipeline = TrainingPipeline(data_path, config_path)
    pipeline.run()
