import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any, Tuple, List, Callable, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import uuid
from functools import wraps

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
        artifacts =
        for root, _, files in os.walk(self.base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                # If run_id is specified, only include artifacts containing that run_id
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts


# custom_mlops/core/step.py

def node(name: str = None, stage: int = None, dependencies: List[str] = None):
    """Decorator to mark a function as a step in a pipeline with rich metadata."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"Executing step: {name or func.__name__}")
            # Add logic for artifact tracking, versioning, etc.
            result = func(*args, **kwargs)
            logging.info(f"Completed step: {name or func.__name__} with result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result

        # Add metadata to the function
        wrapper._is_node = True  # Mark as a node for discovery
        wrapper._node_metadata = {
            "name": name or func.__name__,
            "stage": stage or 0,
            "dependencies": dependencies or
        }
        return wrapper
    return decorator

# custom_mlops/core/pipeline.py
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

        # Execute the pipeline function, which should contain calls to decorated nodes
        pipeline_instance = self.pipeline_func(*self.pipeline_args, **self.pipeline_kwargs)

        logging.info(f"Completed pipeline execution with run_id: {self.run_id}")
        return self.run_id

def stack(func):
    """Decorator to mark a function as a pipeline."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        config_param = next((arg for arg in args if isinstance(arg, Config)), None)
        config_dict = config_param.config_dict if config_param else {}
        stack_instance = Stack(name, config_dict)
        # We need to set the artifact store here as well
        artifact_store_param = next((arg for arg in args if isinstance(arg, ArtifactStore)), None)
        if artifact_store_param:
            stack_instance.set_artifact_store(artifact_store_param)
        stack_instance.set_pipeline(func, *args, **kwargs)
        return stack_instance
    return wrapper


class DataIngestion:
    """Handle data ingestion operations."""

    @node(name="data_ingestion")
    def data_ingestion(self, path: str, config: Config, artifact_store: ArtifactStore) -> Dict[str, pd.DataFrame]:
        """Load and split data into train and test sets."""
        # Define paths for artifacts
        raw_path = config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = config.get("filenames", {}).get("raw_test", "test_data.csv")

        # Load raw data
        df = pd.read_csv(path)

        # Split data
        test_size = config.get("base", {}).get("test_size", 0.2)
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

    @node(name="process_data")
    def process_data(self, data: Dict[str, pd.DataFrame], config: Config, artifact_store: ArtifactStore) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process the train and test data."""
        train_data = data["train_data"]
        test_data = data["test_data"]

        # Define paths for artifacts
        processed_path = config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = config.get("filenames", {}).get("processed_train", "processed_train.csv")
        processed_test_filename = config.get("filenames", {}).get("processed_test", "processed_test.csv")

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
        return {"processed_train": processed_train, "processed_test": processed_test}

@stack
def training_pipeline(data_path: str, config: Config, artifact_store: ArtifactStore):
    """Main pipeline class that orchestrates the training workflow."""

    # Initialize components
    data_ingestion = DataIngestion()
    data_processor = DataProcessor()

    # Execute pipeline stages
    raw_data = data_ingestion.data_ingestion(
        path=data_path, config=config, artifact_store=artifact_store
    )

    processed_data = data_processor.process_data(
        data=raw_data, config=config, artifact_store=artifact_store
    )

    # You can add more steps here, e.g., model training, evaluation, etc.
    return processed_data

if __name__ == "__main__":
    # Create dummy data.csv
    data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)

    # Create dummy config/config.yml
    os.makedirs("config", exist_ok=True)
    config_data = {
        "folder_path": {
            "artifacts": "my_artifacts",
            "raw_data": "raw",
            "processed_data": "processed"
        },
        "filenames": {
            "raw_train": "raw_train.csv",
            "raw_test": "raw_test.csv",
            "processed_train": "processed_train.csv",
            "processed_test": "processed_test.csv"
        },
        "base": {
            "test_size": 0.2
        }
    }
    with open("config/config.yml", "w") as f:
        yaml.dump(config_data, f)

    config_path = "config/config.yml"
    data_path = "data.csv"

    config = Config.load_file(config_path)
    artifact_store = ArtifactStore(config)

    pipeline_instance = training_pipeline(data_path=data_path, config=config, artifact_store=artifact_store)
    run_id = pipeline_instance.run()

    print(f"Pipeline run ID: {run_id}")
    print("\nListing artifacts:")
    for artifact_uri in artifact_store.list_artifacts(run_id=run_id):
        print(f"- {artifact_uri}")


# mlops_framework.py
import os
import logging
import json
import yaml
import pickle
from typing import Dict, Any, Tuple, List, Callable, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import uuid
from functools import wraps
from abc import ABC, abstractmethod

# Configure logging (as before)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Config, ArtifactStore (as before)
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
        artifacts =
        for root, _, files in os.walk(self.base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                # If run_id is specified, only include artifacts containing that run_id
                if run_id is None or run_id in artifact_path:
                    artifacts.append(artifact_path)
        return artifacts

# custom_mlops/core/step.py (as updated before)
def node(name: str = None, stage: int = None, dependencies: List[str] = None, inputs: Dict[str, str] = None, outputs: List[str] = None):
    """Decorator to mark a function as a step in a pipeline with rich metadata."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"Executing step: {name or func.__name__}")
            # Add logic for artifact tracking, versioning, etc.
            result = func(*args, **kwargs)
            logging.info(f"Completed step: {name or func.__name__} with result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result

        # Add metadata to the function
        wrapper._is_node = True  # Mark as a node for discovery
        wrapper._node_metadata = {
            "name": name or func.__name__,
            "stage": stage or 0,
            "dependencies": dependencies or,
            "inputs": inputs or {}, # Map input argument names to the names of output artifacts from dependencies
            "outputs": outputs or [func.__name__] # Default output name is the function name
        }
        return wrapper
    return decorator

# custom_mlops/core/pipeline.py (as updated before)
class Stack:
    """A stack that brings together all components needed to run pipelines and manages their execution."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.artifact_store = None
        self.experiment_tracker = None
        self.model_registry = None
        self.run_id = None
        self.pipeline_func = None
        self.pipeline_args = ()
        self.pipeline_kwargs = {}
        self.nodes = {}
        self.executed_outputs = {}

    def set_artifact_store(self, artifact_store):
        """Set the artifact store for the stack."""
        self.artifact_store = artifact_store
        return self

    def set_experiment_tracker(self, experiment_tracker):
        """Set the experiment tracker for the stack."""
        self.experiment_tracker = experiment_tracker
        return self

    def set_model_registry(self, model_registry):
        """Set the model registry for the stack."""
        self.model_registry = model_registry
        return self

    def set_pipeline(self, func: Callable, *args, **kwargs):
        """Set the pipeline function and its arguments."""
        self.pipeline_func = func
        self.pipeline_args = args
        self.pipeline_kwargs = kwargs
        return self

    def _discover_nodes(self):
        # This part needs a proper implementation to introspect the pipeline function
        # and find decorated nodes. For this example, we'll rely on manual calls.
        pass

    def run(self, run_id=None):
        """Execute the pipeline with the given run_id."""
        self.run_id = run_id or str(uuid.uuid4())
        logging.info(f"Starting pipeline execution with run_id: {self.run_id}")

        if self.pipeline_func is None:
            raise ValueError("Pipeline function is not set.")

        # For this sophisticated orchestration, we'd need to:
        # 1. Discover all nodes in the pipeline.
        # 2. Build a DAG based on the dependencies defined in @node.
        # 3. Execute nodes in topological order, passing outputs of upstream nodes as inputs.

        # This is a complex implementation and beyond a direct integration.
        # For now, we'll stick to the manual execution within the pipeline function.

        pipeline_instance = self.pipeline_func(
            *self.pipeline_args,
            config=self.config,
            artifact_store=self.artifact_store,
            experiment_tracker=self.experiment_tracker,
            model_registry=self.model_registry,
            **self.pipeline_kwargs
        )

        logging.info(f"Completed pipeline execution with run_id: {self.run_id}")
        return self.run_id

def stack(func):
    """Decorator to mark a function as a pipeline."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        config_param = next((arg for arg in args if isinstance(arg, Config)), None)
        config_dict = config_param.config_dict if config_param else {}
        stack_instance = Stack(name, config_dict)

        artifact_store_param = kwargs.get("artifact_store")
        if artifact_store_param:
            stack_instance.set_artifact_store(artifact_store_param)

        experiment_tracker_param = kwargs.get("experiment_tracker")
        if experiment_tracker_param:
            stack_instance.set_experiment_tracker(experiment_tracker_param)

        model_registry_param = kwargs.get("model_registry")
        if model_registry_param:
            stack_instance.set_model_registry(model_registry_param)

        stack_instance.set_pipeline(func, *args, **kwargs)
        return stack_instance
    return wrapper

# custom_mlops/tracking/experiment_tracker.py
class ExperimentTracker(ABC):
    @abstractmethod
    def init_run(self, run_name: str):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metric(self, metric_name: str, value: float):
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        pass

    @abstractmethod
    def end_run(self):
        pass

# custom_mlops/tracking/model_registry.py
class ModelRegistry(ABC):
    @abstractmethod
    def register_model(self, model: Any, model_name: str, version: str = None, metadata: Dict[str, Any] = None):
        pass

    @abstractmethod
    def load_model(self, model_name: str, version: str = None):
        pass

# Example Dummy Implementations
class DummyExperimentTracker(ExperimentTracker):
    def init_run(self, run_name: str):
        logging.info(f"Experiment Tracker: Initializing run '{run_name}'")

    def log_params(self, params: Dict[str, Any]):
        logging.info(f"Experiment Tracker: Logging parameters: {params}")

    def log_metric(self, metric_name: str, value: float):
        logging.info(f"Experiment Tracker: Logging metric '{metric_name}': {value}")

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        logging.info(f"Experiment Tracker: Logging artifact '{artifact_name or artifact_path}' from '{artifact_path}'")

    def end_run(self):
        logging.info("Experiment Tracker: Ending run")

class DummyModelRegistry(ModelRegistry):
    def register_model(self, model: Any, model_name: str, version: str = None, metadata: Dict[str, Any] = None):
        logging.info(f"Model Registry: Registering model '{model_name}' (version: {version}, metadata: {metadata})")

    def load_model(self, model_name: str, version: str = None):
        logging.info(f"Model Registry: Loading model '{model_name}' (version: {version})")
        return None

class DataIngestion:
    """Handle data ingestion operations."""

    @node(name="data_ingestion", outputs=["raw_data"])
    def data_ingestion(self, path: str, config: Config, artifact_store: ArtifactStore) -> Dict[str, pd.DataFrame]:
        # ... (rest of the data_ingestion method as before)
        raw_path = config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = config.get("filenames", {}).get("raw_test", "test_data.csv")
        train_data = pd.read_csv(artifact_store.save_artifact(pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), raw_path, raw_train_filename))
        test_data = pd.read_csv(artifact_store.save_artifact(pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}), raw_path, raw_test_filename))
        return {"raw_data": {"train": train_data, "test": test_data}}

class DataProcessor:
    """Handle data processing operations."""

    @node(name="process_data", inputs={"raw_data": "raw_data"}, outputs=["processed_data"])
    def process_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]], config: Config, artifact_store: ArtifactStore) -> Dict[str, pd.DataFrame]:
        train_data = raw_data["raw_data"]["train"]
        test_data = raw_data["raw_data"]["test"]
        # ... (rest of the process_data method as before)
        processed_path = config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = config.get("filenames", {}).get("processed_train", "processed_train.csv")
        processed_test_filename = config.get("filenames", {}).get("processed_test", "processed_test.csv")
        processed_train = train_data  # Placeholder processing
        processed_test = test_data  # Placeholder processing
        artifact_store.save_artifact(processed_train, processed_path, processed_train_filename)
        artifact_store.save_artifact(processed_test, processed_path, processed_test_filename)
        return {"processed_data": {"train": processed_train, "test": processed_test}}

@stack
def training_pipeline(data_path: str, config: Config, artifact_store: ArtifactStore, experiment_tracker: Optional[ExperimentTracker] = None, model_registry: Optional[ModelRegistry] = None):
    """Main pipeline class that orchestrates the training workflow."""
    if experiment_tracker:
        experiment_tracker.init_run("training_pipeline_run")
        experiment_tracker.log_params({"data_path": data_path, "config_path": config.config_dict})

    # Initialize components
    data_ingestion = DataIngestion()
    data_processor = DataProcessor()

    # Execute pipeline stages (manual for now)
    raw_data = data_ingestion.data_ingestion(
        path=data_path, config=config, artifact_store=artifact_store
    )

    processed_data = data_processor.process_data(
        raw_data=raw_data, config=config, artifact_store=artifact_store
    )

    if experiment_tracker:
        experiment_tracker.log_metric("some_metric", 0.95)
        # Example of logging an artifact
        for artifact_uri in artifact_store.list_artifacts():
            experiment_tracker.log_artifact(artifact_uri)
        experiment_tracker.end_run()

    if model_registry:
        # Example of registering a dummy model
        model_registry.register_model("dummy_model", "MyModel", version="1.0", metadata={"accuracy": 0.95})

    return processed_data

if __name__ == "__main__":
    # Create dummy data and config (as before)
    data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)

    os.makedirs("config", exist_ok=True)
    config_data = {
        "folder_path": {
            "artifacts": "my_artifacts",
            "raw_data": "raw",
            "processed_data": "processed"
        },
        "filenames": {
            "raw_train": "raw_train.csv",
            "raw_test": "raw_test.csv",
            "processed_train": "processed_train.csv",
            "processed_test": "processed_test.csv"
        },
        "base": {
            "test_size": 0.2
        }
    }
    with open("config/config.yml", "w") as f:
        yaml.dump(config_data, f)

    config_path = "config/config.yml"
    data_path = "data.csv"

    config = Config.load_file(config_path)
    artifact_store = ArtifactStore(config)
    experiment_tracker = DummyExperimentTracker()
    model_registry = DummyModelRegistry()

    pipeline_instance = training_pipeline(
        data_path=data_path,
        config=config,
        artifact_store=artifact_store,
        experiment_tracker=experiment_tracker,
        model_registry=model_registry
    )
    run_id = pipeline_instance.run()

    print(f"Pipeline run ID: {run_id}")
    print("\nListing artifacts:")
    for artifact_uri in artifact_store.list_artifacts(run_id=run_id):
        print(f"- {artifact_uri}"