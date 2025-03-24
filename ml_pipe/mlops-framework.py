import os
import logging
import json
import yaml
import pickle
import uuid
from typing import Dict, Any, Tuple, List, Optional, Callable
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        """Retrieve a configuration value with a default."""
        keys = key.split('.')
        value = self.config_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, {})
            else:
                return default
        return value if value else default

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
            logging.error(f"Error loading config file {config_path}: {e}")
            raise ValueError(f"Error loading config file {config_path}: {e}")

class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.base_path = self.config.get("folder_path.artifacts", "artifacts")
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

        try:
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
        except Exception as e:
            logging.error(f"Error saving artifact {name}: {e}")
            raise

    def load_artifact(
        self,
        subdir: str,
        name: str,
    ):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        try:
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
        except Exception as e:
            logging.error(f"Error loading artifact {name}: {e}")
            raise

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

def node(name: str = None, stage: int = None, dependencies: List[str] = None):
    """Decorator to mark a function as a step in a pipeline with rich metadata."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"Executing step: {func.__name__}")
            result = func(*args, **kwargs)
            logging.info(f"Completed step: {func.__name__}")
            return result
        
        # Add metadata to the function
        wrapper._is_node = True  # Mark as a node for discovery
        wrapper._node_metadata = {
            "name": name or func.__name__,
            "stage": stage or 0,
            "dependencies": dependencies or []
        }
        return wrapper
    return decorator

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
        self.run_details = {}
    
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
        start_time = pd.Timestamp.now()
        self.run_id = run_id or str(uuid.uuid4())
        logging.info(f"Starting pipeline execution with run_id: {self.run_id}")
        
        if self.pipeline_func is None:
            raise ValueError("Pipeline function is not set.")
        
        try:
            # Execute the pipeline function
            result = self.pipeline_func(*self.pipeline_args, **self.pipeline_kwargs)
            
            # Store run details
            end_time = pd.Timestamp.now()
            self.run_details = {
                "run_id": self.run_id,
                "pipeline_name": self.name,
                "status": "completed",
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": (end_time - start_time).total_seconds(),
                "tasks": [self.pipeline_func.__name__]
            }
            
            logging.info(f"Completed pipeline execution with run_id: {self.run_id}")
            return result
        except Exception as e:
            # Update run details for failure
            end_time = pd.Timestamp.now()
            self.run_details = {
                "run_id": self.run_id,
                "pipeline_name": self.name,
                "status": "failed",
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": (end_time - start_time).total_seconds(),
                "error": str(e)
            }
            logging.error(f"Pipeline execution failed: {e}")
            raise
    
    def get_run_details(self, run_id=None):
        """Retrieve details of the pipeline run."""
        if run_id and run_id != self.run_id:
            logging.warning(f"Requested run_id {run_id} does not match current run_id")
        return self.run_details

class DataIngestion:
    """Handle data ingestion operations."""
    
    @node(name="data_ingestion", stage=1)
    def data_ingestion(self, path: str, config: Config, artifact_store: ArtifactStore) -> Dict[str, pd.DataFrame]:
        """Load and split data into train and test sets."""
        # Define paths for artifacts
        raw_path = config.get("folder_path.raw_data", "raw_data")
        raw_train_filename = config.get("filenames.raw_train", "train_data.csv")
        raw_test_filename = config.get("filenames.raw_test", "test_data.csv")
        
        # Load raw data
        df = pd.read_csv(path)
        
        # Split data
        test_size = config.get("base.test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        logging.info(
            f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )
        
        # Save raw artifacts
        artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename 
        )
        artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename
        )
        
        logging.info("Data ingestion completed")
        return {
            "train_data": train_data,
            "test_data": test_data
        }

class DataProcessor:
    """Handle data processing operations."""
    
    @node(name="data_processing", stage=2)
    def process_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                     config: Config, artifact_store: ArtifactStore) -> Dict[str, pd.DataFrame]:
        """Process the train and test data."""
        # Define paths for artifacts
        processed_path = config.get("folder_path.processed_data", "processed_data")
        processed_train_filename = config.get("filenames.processed_train", "processed_train.csv")
        processed_test_filename = config.get("filenames.processed_test", "processed_test.csv")
        
        # Implement your data processing logic here
        # This is a placeholder - add your actual processing steps
        processed_train = train_data.copy()
        processed_test = test_data.copy()
        
        # Save processed artifacts
        artifact_store.save_artifact(
            processed_train, subdir=processed_path, name=processed_train_filename
        )
        artifact_store.save_artifact(
            processed_test, subdir=processed_path, name=processed_test_filename
        )
        
        logging.info("Data processing completed")
        return {
            "processed_train": processed_train,
            "processed_test": processed_test
        }

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""
    
    def __init__(self, data_path: str, config_path: str):
        # Load configuration
        self.config = Config.load_file(config_path)
        self.artifact_store = ArtifactStore(self.config)
        
        # Initialize data path
        self.data_path = data_path
        
        # Create stack and set artifact store
        self.stack = Stack("Training Pipeline", self.config.config_dict)
        self.stack.set_artifact_store(self.artifact_store)
        
        # Initialize pipeline components
        self.data_ingestion = DataIngestion()
        self.data_processor = DataProcessor()
    
    @node(name="training_pipeline", stage=0)
    def run(self):
        """Run the complete pipeline."""
        run_id = str(uuid.uuid4())
        logging.info(f"Starting pipeline run with ID: {run_id}")
        
        try:
            # Execute pipeline stages
            ingestion_result = self.data_ingestion.data_ingestion(
                self.data_path, self.config, self.artifact_store
            )
            
            processing_result = self.data_processor.process_data(
                ingestion_result['train_data'], 
                ingestion_result['test_data'], 
                self.config, 
                self.artifact_store
            )
            
            # List artifacts
            artifacts = self.artifact_store.list_artifacts(run_id)
            logging.info(f"Run ID: {run_id}")
            logging.info("Artifacts:")
            for uri in artifacts:
                logging.info(f"- {uri}")
            
            # Get and log run details
            run_details = self.stack.get_run_details()
            logging.info("\nRun Details:")
            logging.info(f"Pipeline: {run_details.get('pipeline_name')}")
            logging.info(f"Status: {run_details.get('status')}")
            logging.info(f"Duration: {run_details.get('duration_seconds', 0):.2f} seconds")
            
            return run_id
        
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise

def main():
    # Example usage
    data_path = "data.csv"
    config_path = "config/config.yml"

    try:
        pipeline = TrainingPipeline(data_path, config_path)
        results = pipeline.run()
        logging.info(f"Pipeline execution results: {results}")
    except Exception as e:
        logging.error(f"Error running pipeline: {e}")

if __name__ == "__main__":
    main()
