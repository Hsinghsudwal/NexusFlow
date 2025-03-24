import os
import uuid
import yaml
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
        elif name.endswith((".json")):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f)
        elif name.endswith((".yml", ".yaml")):
            with open(artifact_path, "w") as f:
                yaml.dump(artifact, f)
        elif isinstance(artifact, str):
            with open(artifact_path, "w") as f:
                f.write(artifact)
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
            elif name.endswith((".json")):
                with open(artifact_path, "r") as f:
                    artifact = json.load(f)
            elif name.endswith((".yml", ".yaml")):
                with open(artifact_path, "r") as f:
                    artifact = yaml.safe_load(f)
            else:
                with open(artifact_path, "r") as f:
                    artifact = f.read()
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
# Task
# -----------------------------

class Task:
    """A task within a pipeline."""
    
    def __init__(self, name: str, func: Callable, description: str = None):
        self.name = name
        self.func = func
        self.description = description or func.__doc__ or ""
        
    def execute(self, context: Dict[str, Any] = None) -> Any:
        """Execute the task function with the given context."""
        logging.info(f"Executing task: {self.name}")
        start_time = datetime.now()
        
        try:
            context = context or {}
            result = self.func(**context)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Task {self.name} completed in {duration:.2f}s")
            
            return result
        except Exception as e:
            logging.error(f"Task {self.name} failed: {str(e)}")
            raise


def task(name: str = None, description: str = None):
    """Decorator to mark a function as a pipeline task."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper.task_info = {
            "name": name or func.__name__,
            "description": description or func.__doc__ or ""
        }
        
        return wrapper
    return decorator


# -----------------------------
# Pipeline
# -----------------------------

class Pipeline:
    """A sequential pipeline of tasks."""
    
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description or ""
        self.tasks = []
        self.task_dependencies = {}  # task_idx -> [dependency_idx, ...]
    
    def add_task(self, task_func: Callable, name: str = None, description: str = None) -> int:
        """Add a task to the pipeline and return its index."""
        # Use task info if decorated with @task
        if hasattr(task_func, 'task_info'):
            name = name or task_func.task_info["name"]
            description = description or task_func.task_info["description"]
        else:
            name = name or task_func.__name__
            description = description or task_func.__doc__ or ""
        
        task = Task(name=name, func=task_func, description=description)
        task_idx = len(self.tasks)
        self.tasks.append(task)
        return task_idx
    
    def add_dependency(self, task_idx: int, depends_on_idx: int) -> None:
        """Add a dependency between tasks."""
        if task_idx >= len(self.tasks) or depends_on_idx >= len(self.tasks):
            raise ValueError("Task index out of range")
        
        if task_idx not in self.task_dependencies:
            self.task_dependencies[task_idx] = []
        
        self.task_dependencies[task_idx].append(depends_on_idx)
    
    def run(self, stack: 'Stack', tags: Optional[Dict[str, Any]] = None) -> str:
        """Run the pipeline with the given stack and return the run ID."""
        run_id = str(uuid.uuid4())
        start_time = datetime.now()
        logging.info(f"Starting pipeline '{self.name}' with run ID: {run_id}")
        
        # Initialize run metadata
        run_metadata = {
            "pipeline_name": self.name,
            "run_id": run_id,
            "start_time": start_time.isoformat(),
            "tags": tags or {},
            "status": "running"
        }
        
        # Get artifact store for this run
        artifact_store = ArtifactStore(stack.config, run_id)
        
        # Save initial run metadata
        artifact_store.save_artifact(run_metadata, "", "run_metadata.json")
        
        try:
            # Set up execution order based on dependencies
            executed_tasks = set()
            task_results = {}
            
            while len(executed_tasks) < len(self.tasks):
                found_executable = False
                
                for idx, task in enumerate(self.tasks):
                    # Skip if already executed
                    if idx in executed_tasks:
                        continue
                    
                    # Check dependencies
                    dependencies = self.task_dependencies.get(idx, [])
                    if all(dep in executed_tasks for dep in dependencies):
                        # Build context with results from dependencies
                        context = {
                            "run_id": run_id,
                            "artifact_store": artifact_store,
                            "stack": stack
                        }
                        
                        # Add results from previous tasks
                        for dep in dependencies:
                            dep_task = self.tasks[dep]
                            context[dep_task.name] = task_results.get(dep)
                        
                        # Execute task
                        result = task.execute(context)
                        task_results[idx] = result
                        executed_tasks.add(idx)
                        found_executable = True
                        
                        # Save task result as artifact
                        artifact_store.save_artifact(result, "results", f"{task.name}.pkl")
                
                if not found_executable and len(executed_tasks) < len(self.tasks):
                    # Circular dependency or other issue
                    raise ValueError("Could not execute all tasks. Check for circular dependencies.")
            
            # All tasks completed successfully
            end_time = datetime.now()
            run_metadata.update({
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "status": "completed"
            })
            
            logging.info(f"Pipeline '{self.name}' completed successfully")
            
        except Exception as e:
            # Pipeline failed
            end_time = datetime.now()
            run_metadata.update({
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "status": "failed",
                "error": str(e)
            })
            
            logging.error(f"Pipeline '{self.name}' failed: {str(e)}")
        
        # Update run metadata
        artifact_store.save_artifact(run_metadata, "", "run_metadata.json")
        
        return run_id


# -----------------------------
# Metrics Tracker
# -----------------------------

class MetricsTracker:
    """Tracks metrics from ML experiments."""
    
    def __init__(self, config: Optional[Config] = None, run_id: str = None):
        self.config = config or Config()
        self.run_id = run_id or "latest"
        self.metrics = {}
        
        # Set up storage path
        base_dir = self.config.get("folder_path", {}).get("metrics", "metrics")
        self.base_path = os.path.join(base_dir, self.run_id)
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Metrics tracker initialized at '{self.base_path}'")
    
    def log_metric(self, name: str, value: float, step: int = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if step is not None:
            metric_entry["step"] = step
        
        self.metrics[name].append(metric_entry)
        
        # Save metrics to disk
        metrics_path = os.path.join(self.base_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f)
        
        logging.info(f"Logged metric '{name}': {value}")
    
    def get_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all tracked metrics."""
        return self.metrics
    
    def get_metric_history(self, name: str) -> List[Dict[str, Any]]:
        """Get history for a specific metric."""
        return self.metrics.get(name, [])
    
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        history = self.get_metric_history(name)
        return history[-1]["value"] if history else None


# -----------------------------
# Model Manager
# -----------------------------

class ModelManager:
    """Manages ML model versions and metadata."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Set up models directory
        base_dir = self.config.get("folder_path", {}).get("models", "models")
        self.models_path = base_dir
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize registry
        registry_path = os.path.join(self.models_path, "registry.json")
        if not os.path.exists(registry_path):
            with open(registry_path, "w") as f:
                json.dump({}, f)
        
        logging.info(f"Model manager initialized at '{self.models_path}'")
    
    def save_model(self, model: Any, name: str, version: str = None, 
                   metadata: Dict[str, Any] = None) -> str:
        """Save a model with metadata."""
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        model_dir = os.path.join(self.models_path, name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        model_metadata = metadata or {}
        model_metadata.update({
            "created_at": datetime.now().isoformat(),
            "name": name,
            "version": version
        })
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f)
        
        # Update registry
        registry_path = os.path.join(self.models_path, "registry.json")
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        if name not in registry:
            registry[name] = {"versions": []}
        
        registry[name]["versions"].append({
            "version": version,
            "path": model_path,
            "metadata_path": metadata_path,
            "created_at": model_metadata["created_at"]
        })
        
        with open(registry_path, "w") as f:
            json.dump(registry, f)
        
        logging.info(f"Model '{name}' version '{version}' saved at {model_path}")
        return version
    
    def load_model(self, name: str, version: str = "latest") -> Any:
        """Load a model by name and version."""
        # Get version from registry if "latest"
        if version == "latest":
            registry_path = os.path.join(self.models_path, "registry.json")
            with open(registry_path, "r") as f:
                registry = json.load(f)
            
            if name not in registry or not registry[name]["versions"]:
                raise ValueError(f"No versions found for model '{name}'")
            
            # Get latest version by creation time
            versions = registry[name]["versions"]
            version = sorted(versions, key=lambda x: x["created_at"], reverse=True)[0]["version"]
        
        # Load model file
        model_path = os.path.join(self.models_path, name, version, "model.pkl")
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{name}' version '{version}' not found")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        logging.info(f"Loaded model '{name}' version '{version}' from {model_path}")
        return model
    
    def get_metadata(self, name: str, version: str = "latest") -> Dict[str, Any]:
        """Get metadata for a model version."""
        # Get version from registry if "latest"
        if version == "latest":
            registry_path = os.path.join(self.models_path, "registry.json")
            with open(registry_path, "r") as f:
                registry = json.load(f)
            
            if name not in registry or not registry[name]["versions"]:
                raise ValueError(f"No versions found for model '{name}'")
            
            # Get latest version by creation time
            versions = registry[name]["versions"]
            version = sorted(versions, key=lambda x: x["created_at"], reverse=True)[0]["version"]
        
        # Load metadata file
        metadata_path = os.path.join(self.models_path, name, version, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata for model '{name}' version '{version}' not found")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions."""
        registry_path = os.path.join(self.models_path, "registry.json")
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        result = {}
        for name, info in registry.items():
            versions = [v["version"] for v in info["versions"]]
            result[name] = versions
        
        return result
    
    def compare_models(self, name: str, version1: str, version2: str, 
                       metric: str = None) -> Dict[str, Any]:
        """Compare two model versions."""
        metadata1 = self.get_metadata(name, version1)
        metadata2 = self.get_metadata(name, version2)
        
        if metric and "metrics" in metadata1 and "metrics" in metadata2:
            comparison = {
                "model": name,
                "metric": metric,
                "versions": {
                    version1: metadata1["metrics"].get(metric),
                    version2: metadata2["metrics"].get(metric)
                },
                "diff": metadata2["metrics"].get(metric, 0) - metadata1["metrics"].get(metric, 0)
            }
        else:
            comparison = {
                "model": name,
                "versions": [version1, version2],
                "metadata1": metadata1,
                "metadata2": metadata2
            }
        
        return comparison


# -----------------------------
# Data Manager
# -----------------------------

class DataManager:
    """Manages datasets and preprocessing."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Set up data directory
        base_dir = self.config.get("folder_path", {}).get("data", "data")
        self.data_path = base_dir
        os.makedirs(self.data_path, exist_ok=True)
        
        logging.info(f"Data manager initialized at '{self.data_path}'")
    
    def save_dataset(self, data: Any, name: str, version: str = None, 
                     metadata: Dict[str, Any] = None) -> str:
        """Save a dataset with metadata."""
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        data_dir = os.path.join(self.data_path, name, version)
        os.makedirs(data_dir, exist_ok=True)
        
        # Save dataset
        data_path = os.path.join(data_dir, "data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        
        # Save metadata
        data_metadata = metadata or {}
        data_metadata.update({
            "created_at": datetime.now().isoformat(),
            "name": name,
            "version": version
        })
        
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(data_metadata, f)
        
        logging.info(f"Dataset '{name}' version '{version}' saved at {data_path}")
        return version
    
    def load_dataset(self, name: str, version: str = "latest") -> Any:
        """Load a dataset by name and version."""
        # Get latest version if not specified
        if version == "latest":
            versions = self.list_dataset_versions(name)
            if not versions:
                raise ValueError(f"No versions found for dataset '{name}'")
            version = versions[0]
        
        # Load dataset file
        data_path = os.path.join(self.data_path, name, version, "data.pkl")
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset '{name}' version '{version}' not found")
        
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        logging.info(f"Loaded dataset '{name}' version '{version}' from {data_path}")
        return data
    
    def list_dataset_versions(self, name: str) -> List[str]:
        """List all versions of a dataset."""
        dataset_dir = os.path.join(self.data_path, name)
        if not os.path.exists(dataset_dir):
            return []
        
        versions = sorted([v for v in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, v))],
                           reverse=True)
        
        return versions
    
    def list_datasets(self) -> Dict[str, List[str]]:
        """List all datasets and their versions."""
        if not os.path.exists(self.data_path):
            return {}
        
        result = {}
        for name in os.listdir(self.data_path):
            dataset_dir = os.path.join(self.data_path, name)
            if os.path.isdir(dataset_dir):
                versions = self.list_dataset_versions(name)
                result[name] = versions
        
        return result


# -----------------------------
# Stack
# -----------------------------

class Stack:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = Config(config or {})
        
        # Initialize components
        self.artifact_store = ArtifactStore(self.config)
        self.metrics_tracker = MetricsTracker(self.config)
        self.model_manager = ModelManager(self.config)
        self.data_manager = DataManager(self.config)
        
        logging.info(f"Stack '{name}' initialized")
    
    def run_pipeline(self, pipeline: Pipeline, tags: Optional[Dict[str, Any]] = None) -> str:
        """Run a pipeline with this stack."""
        return pipeline.run(self, tags=tags)
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Get information about a specific run."""
        artifact_store = ArtifactStore(self.config, run_id)
        metadata = artifact_store.load_artifact("", "run_metadata.json")
        
        if metadata is None:
            raise ValueError(f"No metadata found for run '{run_id}'")
        
        return metadata
    
    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent pipeline runs."""
        base_dir = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        if not os.path.exists(base_dir):
            return []
        
        runs = []
        for run_id in os.listdir(base_dir):
            metadata_path = os.path.join(base_dir, run_id, "run_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    runs.append(metadata)
        
        # Sort by start time (newest first) and limit
        sorted_runs = sorted(runs, key=lambda x: x.get("start_time", ""), reverse=True)
        return sorted_runs[:limit]


# -----------------------------
# Example Usage
# -----------------------------

def create_example_ml_pipeline():
    """Create an example ML pipeline for demonstration."""
    # Create the pipeline
    pipeline = Pipeline(name="example_ml_pipeline", description="A simple ML pipeline example")
    
    # Define tasks
    @task(name="load_data", description="Load training data")
    def load_data(run_id, artifact_store, stack, **kwargs):
        """Load and prepare the dataset."""
        import numpy as np
        
        # Generate synthetic data
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        
        # Store as artifact
        data = {"X": X, "y": y}
        artifact_store.save_artifact(data, "data", "training_data.pkl")
        
        # Save in data manager
        stack.data_manager.save_dataset(
            data, 
            name="example_dataset",
            metadata={"samples": len(y), "features": X.shape[1]}
        )
        
        # Log data metrics
        stack.metrics_tracker.log_metric("data_size", len(y))
        stack.metrics_tracker.log_metric("feature_count", X.shape[1])
        
        return data
    
    @task(name="train_model", description="Train ML model")
    def train_model(run_id, artifact_store, stack, load_data, **kwargs):
        """Train a model on the data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Get data from previous step
        data = load_data
        X, y = data["X"], data["y"]
        
        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Evaluate on training data
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Log metrics
        stack.metrics_tracker.log_metric("train_accuracy", accuracy)
        
        # Save model
        stack.model_manager.save_model(
            model,
            name="example_model",
            metadata={"accuracy": accuracy, "algorithm": "RandomForestClassifier"}
        )
        
        # Save as artifact
        artifact_store.save_artifact(model, "models", "trained_model.pkl")
        
        return model
    
    @task(name="evaluate_model", description="Evaluate ML model")
    def evaluate_model(run_id, artifact_store, stack, load_data, train_model, **kwargs):
        """Evaluate the trained model."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Get data and model
        data = load_data
        model = train_model
        X, y = data["X"], data["y"]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0)
        }
        
        # Log metrics
        for name, value in metrics.items():
            stack.metrics_tracker.log_metric(name, value)
        
        # Save metrics as artifact
        artifact_store.save_artifact(metrics, "evaluation", "metrics.json")
        
        return metrics
    
    # Add tasks to the pipeline
    load_data_idx = pipeline.add_task(load_data)
    train_model_idx = pipeline.add_task(train_model)
    evaluate_model_idx = pipeline.add_task(evaluate_model)
    
    # Set dependencies
    pipeline.add_dependency(train_model_idx, load_data_idx)
    pipeline.add_dependency(evaluate_model_idx, load_data_idx)
    pipeline.add_dependency(evaluate_model_idx, train_model_idx)
    
    return pipeline


def run_example():
    """Run the example ML pipeline."""
    # Create configuration
    config = {
        "folder_path": {
            "artifacts": "./artifacts",
            "metrics": "./metrics",
            "models": "./models",
            "data": "./data"
        }
    }
    
    # Initialize stack
    stack = Stack(name="ml_stack", config=config)
    
    # Create and run pipeline
    pipeline = create_example_ml_pipeline()
    run_id = stack.run_pipeline(pipeline, tags={"env": "development", "purpose": "demo"})
    
    # Display results
    print(f"Pipeline run completed with ID: {run_id}")
    
    # List artifacts
    artifacts = stack.artifact_store.list_artifacts(run_id)
    print("\nArtifacts:")
    for artifact in artifacts:
        print(f"- {artifact}")
    
    # Get run info
    run_info = stack.get_run_info(run_id)
    print("\nRun Info:")
    print(f"Pipeline: {run_info['pipeline_name']}")
    print(f"Status: {run_info['status']}")
    print(f"Duration: {run_info.get('duration', 'N/A')} seconds")
    
    # List models
    models = stack.model_manager.list_models()
    print("\nMo