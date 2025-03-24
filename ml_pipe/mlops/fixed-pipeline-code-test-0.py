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


# Base task class
class PipelineTask:
    """Base class for all pipeline tasks."""
    
    def __init__(self, name):
        self.name = name
        
    def execute(self, inputs=None):
        """Execute the task and return outputs."""
        raise NotImplementedError("Task execution must be implemented by subclasses")


# ----------------------------- 
# Data Ingestion
# -----------------------------

class DataIngestionTask(PipelineTask):
    def __init__(self, config, artifact_store):
        super().__init__(name="data_ingestion")
        self.config = config
        self.artifact_store = artifact_store

    def execute(self, inputs=None):
        """Execute data ingestion and return train/test data paths."""
        data_path = inputs.get('data_path')
        if not data_path:
            raise ValueError("Data path not provided for ingestion task")
            
        # Define paths and filenames
        raw_path = "raw"
        raw_train_filename = "train_data.csv"
        raw_test_filename = "test_data.csv"
      
        # Load raw data
        logging.info(f"Loading data from {data_path}")
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
        
        # Return the paths so they can be used as inputs for the next task
        return {
            'train_data_path': train_path,
            'test_data_path': test_path,
            'train_data': train_data,
            'test_data': test_data
        }


# ----------------------------- 
# Data Processing
# -----------------------------

class DataProcessingTask(PipelineTask):
    def __init__(self, config, artifact_store):
        super().__init__(name="data_processing")
        self.config = config
        self.artifact_store = artifact_store

    def execute(self, inputs=None):
        """Process the data from previous task."""
        if not inputs or 'train_data' not in inputs:
            raise ValueError("Required input 'train_data' not provided to DataProcessingTask")
            
        train_data = inputs['train_data']
        test_data = inputs['test_data']
        
        # Example processing: Drop na values and scale features
        logging.info("Processing data...")
        processed_train = train_data.dropna()
        processed_test = test_data.dropna()
        
        # Example: Additional processing steps could go here
        
        # Save processed artifacts
        processed_path = "processed"
        processed_train_filename = "processed_train_data.csv"
        processed_test_filename = "processed_test_data.csv"
        
        train_path = self.artifact_store.save_artifact(
            processed_train, subdir=processed_path, name=processed_train_filename
        )
        test_path = self.artifact_store.save_artifact(
            processed_test, subdir=processed_path, name=processed_test_filename
        )
        
        logging.info("Data processing completed")
        
        return {
            'processed_train_path': train_path,
            'processed_test_path': test_path,
            'processed_train': processed_train,
            'processed_test': processed_test
        }


# ----------------------------- 
# Stack Pipeline
# -----------------------------

class StackPipeline:
    """Pipeline that manages execution of a sequence of tasks."""
    
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.tasks = []
        self.artifact_store = ArtifactStore(config)
        self.run_history = {}
        
    def add_task(self, task):
        """Add a task to the pipeline."""
        if not isinstance(task, PipelineTask):
            raise TypeError("Task must be an instance of PipelineTask")
        self.tasks.append(task)
        logging.info(f"Added task '{task.name}' to pipeline '{self.name}'")
        
    def run(self):
        """Execute all tasks in sequence, passing outputs as inputs to the next task."""
        import time
        import uuid
        
        run_id = str(uuid.uuid4())[:8]  # Generate a unique run ID
        start_time = time.time()
        
        logging.info(f"Starting pipeline '{self.name}' with run ID: {run_id}")
        
        # Initialize results dictionary
        results = {'data_path': self.config.get('data_path')}
        
        try:
            # Execute each task in sequence
            for i, task in enumerate(self.tasks):
                logging.info(f"Executing task {i+1}/{len(self.tasks)}: '{task.name}'")
                task_start = time.time()
                
                # Execute the task with the results from previous tasks
                task_results = task.execute(results)
                task_duration = time.time() - task_start
                
                # Update results with the outputs from this task
                if task_results:
                    results.update(task_results)
                
                logging.info(f"Task '{task.name}' completed in {task_duration:.2f} seconds")
                
            # Record run details
            end_time = time.time()
            duration = end_time - start_time
            
            self.run_history[run_id] = {
                'pipeline_name': self.name,
                'status': 'completed',
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration,
                'tasks': [task.name for task in self.tasks]
            }
            
            logging.info(f"Pipeline '{self.name}' completed in {duration:.2f} seconds")
            return run_id
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            self.run_history[run_id] = {
                'pipeline_name': self.name,
                'status': 'failed',
                'error': str(e)
            }
            raise
            
    def get_run_details(self, run_id):
        """Get details for a specific pipeline run."""
        return self.run_history.get(run_id, {})


# ----------------------------- 
# Training Pipeline
# -----------------------------

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, data_path: str, config_path: str = "config/config.yml"):
        self.data_path = data_path
        self.config = Config.load_file(config_path)
        
    def run(self) -> Tuple[str, StackPipeline]:
        """Execute the training pipeline."""
        # Create stack
        pipe = StackPipeline(name="training_pipeline", config=self.config.config_dict)
        
        # Set data path in config
        pipe.config['data_path'] = self.data_path
        
        # Initialize artifact store
        artifact_store = pipe.artifact_store
        
        # Create task instances
        data_ingestion_task = DataIngestionTask(self.config, artifact_store)
        data_processing_task = DataProcessingTask(self.config, artifact_store)
        
        # Add tasks to pipeline
        pipe.add_task(data_ingestion_task)
        pipe.add_task(data_processing_task)
        # Add more tasks as needed:
        # pipe.add_task(ModelTrainingTask(self.config, artifact_store))
        # pipe.add_task(ModelEvaluationTask(self.config, artifact_store))
        
        # Run the pipeline
        try:
            run_id = pipe.run()
            logging.info(f"Pipeline completed successfully with run ID: {run_id}")
            
            # Output run summary
            self._print_run_summary(pipe, run_id)
            
            return run_id, pipe
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise
            
    def _print_run_summary(self, pipe: StackPipeline, run_id: str) -> None:
        """Print a summary of the pipeline run."""
        # List artifacts
        artifacts = pipe.artifact_store.list_artifacts()
        print(f"\nRun ID: {run_id}")
        print("\nArtifacts:")
        for uri in artifacts:
            print(f"- {uri}")
            
        # Get run details
        run_details = pipe.get_run_details(run_id)
        print("\nRun Details:")
        print(f"Pipeline: {run_details.get('pipeline_name')}")
        print(f"Status: {run_details.get('status')}")
        print(f"Duration: {run_details.get('duration_seconds', 0):.2f} seconds")
        print(f"Tasks executed: {', '.join(run_details.get('tasks', []))}")
        
        # Check if run was successful
        if run_details.get("status") == "completed":
            print("Pipeline completed successfully")


# ----------------------------- 
# Example Usage
# ----------------------------- 

def main():
    """Main entry point for running the pipeline."""
    # Path to your data file
    data_path = "data.csv"
    
    # Create and run the pipeline
    try:
        pipeline = TrainingPipeline(data_path)
        run_id, pipe = pipeline.run()
        
        print("\n" + "="*50)
        print("Pipeline execution complete!")
        print(f"Run ID: {run_id}")
        print("="*50)
        
        return run_id, pipe
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        return None, None


if __name__ == "__main__":
    main()
