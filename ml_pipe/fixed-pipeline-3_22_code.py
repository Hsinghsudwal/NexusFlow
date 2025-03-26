import os
import logging
import json
import yaml
import pickle
import uuid
from typing import Dict, Any, Tuple, List, Callable
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
        run_id: str = None,
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
        elif name.endswith(".json"):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, indent=2)
        else:
            raise ValueError(f"Unsupported format for {name}")
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        return artifact_path

    def load_artifact(
        self,
        subdir: str,
        name: str,
        run_id: str = None,
    ):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
            elif name.endswith(".json"):
                with open(artifact_path, "r") as f:
                    artifact = json.load(f)
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


class Task:
    """Represents a single task/node in the pipeline."""
    
    def __init__(self, func: Callable, name: str = None, task_id: int = None):
        self.func = func
        self.name = name or func.__name__
        self.task_id = task_id
        
    def run(self, pipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task function and update the context with its results."""
        logging.info(f"Running task '{self.name}'")
        
        # Record task start in run history
        task_start_time = pd.Timestamp.now()
        task_record = {
            "task_name": self.name,
            "task_id": self.task_id,
            "start_time": task_start_time.isoformat(),
            "status": "running"
        }
        
        if pipeline.current_run_id in pipeline.run_history:
            pipeline.run_history[pipeline.current_run_id]["nodes"].append(task_record)
        
        try:
            # Execute the task function
            result = self.func(context)
            
            # Update task record with success
            task_end_time = pd.Timestamp.now()
            duration = (task_end_time - task_start_time).total_seconds()
            
            task_record.update({
                "status": "completed",
                "end_time": task_end_time.isoformat(),
                "duration_seconds": duration
            })
            
            logging.info(f"Task '{self.name}' completed in {duration:.2f} seconds")
            
            # Update context with task results if they were returned
            if result is not None:
                if isinstance(result, dict):
                    context.update(result)
                else:
                    # If result is not a dict, store it under the task name
                    context[self.name] = result
            
            return context
            
        except Exception as e:
            # Update task record with failure
            task_record.update({
                "status": "failed",
                "error": str(e)
            })
            
            logging.error(f"Task '{self.name}' failed: {e}")
            raise


# Decorator to create task functions
def node(name: str = None, task_id: int = None):
    """Decorator to mark functions as pipeline node."""
    def decorator(func):
        # Attach metadata to the function
        func._task_metadata = {
            "name": name or func.__name__,
            "task_id": task_id
        }
        return func
    
    return decorator


class StackPipeline:
    """Pipeline that manages the execution of a series of nodes."""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = Config(config)
        self.nodes = []
        self.run_history = {}
        self.current_run_id = None
        self.artifact_store = ArtifactStore(self.config)
        logging.info(f"Pipeline '{name}' initialized")
        
    def add_task(self, func: Callable, name: str = None, task_id: int = None) -> None:
        """Add a task function to the pipeline."""
        # Check if func has metadata from the @node decorator
        if hasattr(func, '_task_metadata'):
            metadata = func._task_metadata
            name = name or metadata.get('name')
            task_id = task_id or metadata.get('task_id')
        
        task = Task(func, name=name, task_id=task_id)
        self.nodes.append(task)
        logging.info(f"Task '{task.name}' added to pipeline '{self.name}'")
        
    def run(self, context: Dict[str, Any] = None) -> str:
        """Run all nodes in the pipeline and return the run ID."""
        self.current_run_id = f"run_{uuid.uuid4().hex[:8]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = pd.Timestamp.now()
        
        # Initialize run in history
        self.run_history[self.current_run_id] = {
            "pipeline_name": self.name,
            "start_time": start_time.isoformat(),
            "status": "running",
            "nodes": []
        }
        
        logging.info(f"Starting pipeline '{self.name}' with run ID: {self.current_run_id}")
        
        # Initialize context
        current_context = context or {}
        
        try:
            # Execute each task in sequence
            for task in self.nodes:
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
            
            logging.error(f"Pipeline '{self.name}' failed: {e}")
            
            # Save run details as artifact
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
    """Handles data ingestion tasks in the pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    @node(name="data_loader", task_id=1)
    def data_ingestion(self, context):
        """Load and split data into train and test sets."""
        # Get data path from context
        data_path = context.get("data_path")
        if not data_path:
            raise ValueError("No data_path provided in context")
            
        # Define paths for artifacts
        raw_path = self.config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = self.config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = self.config.get("filenames", {}).get("raw_test", "test_data.csv")
        
        # Load raw data
        df = pd.read_csv(data_path)
        
        # Split data
        test_size = self.config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(
            f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )
        
        # Save raw artifacts
        train_path = self.artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename 
        )
        test_path = self.artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename
        )
        
        logging.info("Data ingestion completed")
        
        # Update context with results
        return {
            "train_data": train_data,
            "test_data": test_data,
            "train_path": train_path,
            "test_path": test_path
        }


class DataProcess:
    """Handles data processing tasks in the pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    @node(name="data_process", task_id=2)
    def data_processing(self, context):
        """Process train and test data."""
        # Get data from context
        train_data = context.get("train_data")
        test_data = context.get("test_data")
        
        if train_data is None or test_data is None:
            raise ValueError("Missing train_data or test_data in context")
            
        # Define paths for artifacts
        process_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        process_train_filename = self.config.get("filenames", {}).get("processed_train", "processed_train.csv")
        process_test_filename = self.config.get("filenames", {}).get("processed_test", "processed_test.csv")
        
        # Example processing logic (replace with your actual processing)
        # For demonstration, we'll just add a simple feature
        train_process = train_data.copy()
        test_process = test_data.copy()
        
        # Add a sample feature (replace with your actual processing)
        if 'feature1' in train_process.columns and 'feature2' in train_process.columns:
            train_process['feature_sum'] = train_process['feature1'] + train_process['feature2']
            test_process['feature_sum'] = test_process['feature1'] + test_process['feature2']
            
        # Save artifacts
        train_proc_path = self.artifact_store.save_artifact(
            train_process, subdir=process_path, name=process_train_filename 
        )
        test_proc_path = self.artifact_store.save_artifact(
            test_process, subdir=process_path, name=process_test_filename
        )
        
        logging.info("Data processing completed")
        
        # Update context with results
        return {
            "processed_train_data": train_process,
            "processed_test_data": test_process,
            "processed_train_path": train_proc_path,
            "processed_test_path": test_proc_path
        }


class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        self.config = Config.load_file(config_path)
        
        # Initialize components
        self.data_ingestion = DataIngestion(self.config)
        self.data_processing = DataProcess(self.config)
        
        # Create pipeline
        self.pipeline = StackPipeline("training_pipeline", config=self.config.config_dict)
        
        # Register tasks
        self._register_tasks()
        
    def _register_tasks(self):
        """Register all tasks with the pipeline."""
        # For data ingestion, we'll create a wrapper to inject parameters from self
        def data_ingestion_task(context):
            # Add the data path to the context
            context["data_path"] = self.data_path
            return self.data_ingestion.data_ingestion(context)
            
        # Add tasks to the pipeline
        self.pipeline.add_task(data_ingestion_task, name="data_ingestion", task_id=1)
        self.pipeline.add_task(self.data_processing.data_processing, name="data_processing", task_id=2)
        
    def run(self):
        """Run the training pipeline."""
        try:
            run_id = self.pipeline.run({})
            logging.info(f"Pipeline completed successfully with run ID: {run_id}")
            return run_id
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    data_path = "data.csv"
    config_path = "config/config.yml"
    
    pipeline = TrainingPipeline(data_path, config_path)
    pipeline.run()
