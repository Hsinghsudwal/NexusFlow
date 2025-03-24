import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any, Tuple, List
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
        self.base_path = self.config.get("folder_path", {}).get(
            "artifacts", "artifacts"
        )
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(
        self,
        artifact: Any,
        subdir: str,
        name: str,
    ) -> str:
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

    def load_artifact(
        self,
        subdir: str,
        name: str,
    ):
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





class Node:
    """Base class for all pipeline tasks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        # self.description = description
        
    def run(self, stack: 'StackPipeline', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task and return updated context."""
        logging.info(f"Starting task: {self.name}")
        start_time = pd.Timestamp.now()
        
        try:
            updated_context = self._run(stack, context)
            
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Task '{self.name}' completed in {duration:.2f} seconds")
            
            # Record task execution in the stack's run history
            task_result = {
                "name": self.name,
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            }
            
            if hasattr(stack, 'current_run_id') and stack.current_run_id:
                if stack.current_run_id not in stack.run_history:
                    stack.run_history[stack.current_run_id] = {"tasks": []}
                stack.run_history[stack.current_run_id]["tasks"].append(task_result)
            
            return updated_context
            
        except Exception as e:
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            logging.error(f"Task '{self.name}' failed after {duration:.2f} seconds: {str(e)}")
            
            # Record failure
            task_result = {
                "name": self.name,
                "status": "failed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": str(e)
            }
            
            if hasattr(stack, 'current_run_id') and stack.current_run_id:
                if stack.current_run_id not in stack.run_history:
                    stack.run_history[stack.current_run_id] = {"tasks": []}
                stack.run_history[stack.current_run_id]["tasks"].append(task_result)
            
            raise
    
    def _run(self, stack: 'StackPipeline', context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of task logic. Should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


# Decorator to create task functions
def node(name: str = None,task: int=None):
    """Decorator to mark functions as pipeline tasks."""
    def decorator(func):
        # We'll keep the original function but add metadata
        func._task_metadata = {
            "name": name or func.__name__,
            "task": task
        }
        return func
    return decorator


class StackPipeline:
    """Pipeline that manages the execution of a series of tasks."""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = Config(config)
        self.tasks = []
        self.run_history = {}
        self.current_run_id = None
        self.artifact_store = ArtifactStore(self.config)
        logging.info(f"Pipeline '{name}' initialized")
        
    def add_task(self, task: Task) -> None:
        """Add a task to the pipeline."""
        self.tasks.append(task)
        logging.info(f"Task '{task.name}' added to pipeline '{self.name}'")
        
    def run(self, context: Dict[str, Any] = None) -> str:
        """Run all tasks in the pipeline and return the run ID."""
        import uuid
        
        self.current_run_id = f"run_{uuid.uuid4().hex[:8]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = pd.Timestamp.now()
        
        # Initialize run in history
        self.run_history[self.current_run_id] = {
            "pipeline_name": self.name,
            "start_time": start_time.isoformat(),
            "status": "running",
            "tasks": []
        }
        
        logging.info(f"Starting pipeline '{self.name}' with run ID: {self.current_run_id}")
        
        # Initialize context
        current_context = context or {}
        
        try:
            # Execute each task in sequence
            for task in self.tasks:
                current_context = task.run(self, current_context)
                
            # Record successful completion
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            self.run_history[self.current_run_id].update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            })
            
            logging.info(f"Pipeline '{self.name}' completed in {duration:.2f} seconds")
            
            # Save run details as artifact
            self.artifact_store.save_artifact(
                self.run_history[self.current_run_id],
                subdir="runs",
                name=f"{self.current_run_id}_details.json",
                run_id=self.current_run_id
            )
            
            return self.current_run_id
            
        except Exception as e:
            # Record failure
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            self.run_history[self.current_run_id].update({
                "status": "failed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": str(e)
            })
            
            logging.error(f"Pipeline '{self.name}' failed after {duration:.2f} seconds: {str(e)}")
            
            # Save run details even for failed runs
            self.artifact_store.save_artifact(
                self.run_history[self.current_run_id],
                subdir="runs",
                name=f"{self.current_run_id}_details.json",
                run_id=self.current_run_id
            )
            
            raise
            
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get details about a specific pipeline run."""
        if run_id in self.run_history:
            return self.run_history[run_id]
        
        # Try to load from artifact store
        run_details = self.artifact_store.load_artifact(
            subdir="runs",
            name=f"{run_id}_details.json",
            run_id=run_id
        )
        
        if run_details:
            # Cache in memory
            self.run_history[run_id] = run_details
            return run_details
            
        logging.warning(f"No details found for run ID: {run_id}")
        return None




class DataIngestion:
    def __init__(self):

        self.artifact_store = ArtifactStore(config)

    @node(name="data_loader", task= 1)
    def data_ingestion(self, path, config):
        # Define paths for artifacts
        raw_path = config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = config.get("filenames", {}).get("raw_test", "test_data.csv")
        
        # Load raw data
        df = pd.read_csv(path)
        # Split data
        test_size = self.config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(
            f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )
        # Save raw artifacts
        self.artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename 
        )
        self.artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename
        )
        logging.info("Data ingestion completed")
        return train_data, test_data

class DataProcess:
    def __init__(self):

        self.artifact_store = ArtifactStore(config)

    @node(name="data_process", task= 2)
    def data_processing(self, trandata,testdata, config):
        # Define paths for artifacts
       
        # process logic
        # Save artifacts
        self.artifact_store.save_artifact(
            train_process, subdir=process_path, name=process_train_filename 
        )
        self.artifact_store.save_artifact(
            test_process, subdir=raw_path, name=process_test_filename
        )
        logging.info("Data process completed")
        return train_process, test_process


# ----------------------------- 
# Training Pipeline
# -----------------------------

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        self.config = config_path

    def run(data_path,config_path):

        dataingest = DataIngestion()
        dataprocess = DataProcess()

        pipeline = StackPipeline("my_pipeline")
        pipeline.add_task(dataingest.data_ingestion(self.data_path,self.config))
        pipeline.add_task(dataprocess.data_processing(dataingest.data_ingestion(self.data_path,self.config),self.config))


if __name__ == "__main__":
    data_path = "data.csv"
    config_path = "config/config.yml"
    
    pipeline = TrainingPipeline(data_path, config_path)
    pipeline.run()