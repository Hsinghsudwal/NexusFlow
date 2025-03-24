import os
import uuid
import yaml
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar
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
        self.run_id = run_id or str(uuid.uuid4())
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
# Metrics Store
# -----------------------------

class MetricsStore:
    """Tracks and stores metrics for pipeline runs."""
    
    def __init__(self, config: Optional[Config] = None, run_id: str = None):
        self.config = config or Config()
        self.run_id = run_id or "latest"
        self._metrics: Dict[str, Dict[str, float]] = {}
        
        # Set up base path for metrics
        base_dir = self.config.get("folder_path", {}).get("metrics", "metrics")
        self.base_path = os.path.join(base_dir, self.run_id)
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Metrics store initialized at '{self.base_path}'")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value, optionally at a specific step."""
        if name not in self._metrics:
            self._metrics[name] = {}
        
        step_key = str(step) if step is not None else "latest"
        self._metrics[name][step_key] = value
        
        # Save metrics to disk immediately for persistence
        self._save_metrics()
        logging.info(f"Logged metric {name}={value} at step {step_key}")
    
    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        metrics_path = os.path.join(self.base_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f)
    
    def get_metric(self, name: str, step: Optional[int] = None) -> Optional[float]:
        """Get a metric value, optionally at a specific step."""
        if name not in self._metrics:
            return None
        
        step_key = str(step) if step is not None else "latest"
        return self._metrics[name].get(step_key)
    
    def get_metrics_history(self, name: str) -> Dict[str, float]:
        """Get all recorded values for a metric across steps."""
        return self._metrics.get(name, {})
    
    def load_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load metrics from disk."""
        metrics_path = os.path.join(self.base_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self._metrics = json.load(f)
        return self._metrics


# -----------------------------
# Pipeline
# -----------------------------

class Pipeline:
    """A sequential pipeline of tasks."""
    
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description or ""
        self.tasks: List[Callable] = []
        self.task_dependencies: Dict[int, List[int]] = {}  # task_idx -> [dependency_idx, ...]
    
    def add_task(self, task_func: Callable) -> int:
        """Add a task to the pipeline and return its index."""
        if not hasattr(task_func, 'task_metadata'):
            raise ValueError("Function must be decorated with @task")
        
        task_idx = len(self.tasks)
        self.tasks.append(task_func)
        return task_idx
    
    def add_dependency(self, task_idx: int, depends_on_idx: int) -> None:
        """Add a dependency between tasks."""
        if task_idx >= len(self.tasks) or depends_on_idx >= len(self.tasks):
            raise ValueError("Task index out of range")
        
        if task_idx not in self.task_dependencies:
            self.task_dependencies[task_idx] = []
        
        self.task_dependencies[task_idx].append(depends_on_idx)
    
    def run(self, stack: 'Stack', tags: Optional[Dict[str, str]] = None) -> str:
        """Run the pipeline with the given stack and return the run ID."""
        run_id = str(uuid.uuid4())
        run_start = datetime.now().isoformat()
        logging.info(f"Starting pipeline '{self.name}' with run ID: {run_id}")
        
        # Initialize run context
        run_context = {
            "run_id": run_id,
            "pipeline_name": self.name,
            "start_time": run_start,
            "tags": tags or {},
            "results": {},
            "status": "running"
        }
        
        # Create an artifact store for this run
        artifact_store = ArtifactStore(stack.config, run_id)
        metrics_store = MetricsStore(stack.config, run_id)
        
        try:
            # Topologically sort tasks based on dependencies
            executed_tasks = set()
            results = {}
            
            while len(executed_tasks) < len(self.tasks):
                for i, task_func in enumerate(self.tasks):
                    if i in executed_tasks:
                        continue
                    
                    # Check if all dependencies are satisfied
                    dependencies = self.task_dependencies.get(i, [])
                    if all(dep in executed_tasks for dep in dependencies):
                        # Execute the task
                        task_name = task_func.task_metadata.name
                        logging.info(f"Executing task {task_name} ({i+1}/{len(self.tasks)})")
                        
                        # Prepare context for the task
                        task_context = {
                            "run_id": run_id,
                            "task_name": task_name,
                            "artifact_store": artifact_store,
                            "metrics_store": metrics_store,
                            "results": results.copy()  # Results from previous tasks
                        }
                        
                        # Execute the task with the context
                        result = task_func(context=task_context)
                        results[task_name] = result
                        executed_tasks.add(i)
                        
                        # Save task result as an artifact
                        artifact_name = f"{task_name}_result.pkl"
                        artifact_store.save_artifact(result, "task_results", artifact_name)
            
            # Update run context with success status
            run_context["end_time"] = datetime.now().isoformat()
            run_context["status"] = "completed"
            logging.info(f"Pipeline '{self.name}' completed successfully")
            
        except Exception as e:
            # Update run context with failure status
            run_context["end_time"] = datetime.now().isoformat()
            run_context["status"] = "failed"
            run_context["error"] = str(e)
            logging.error(f"Pipeline '{self.name}' failed: {str(e)}")
        
        # Save run context as an artifact
        artifact_store.save_artifact(run_context, "", "run_context.json")
        
        # Update stack with the latest run
        stack.latest_run_id = run_id
        
        return run_id


# -----------------------------
# Stack
# -----------------------------

class Stack:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = Config(config or {})
        self.artifact_store = ArtifactStore(self.config)
        self.metrics_store = MetricsStore(self.config)
        self.latest_run_id = None
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if model_path.endswith(".pkl"):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format for {model_path}")
    
    def save_model(self, model: Any, model_path: str) -> str:
        """Save a model to disk."""
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        if model_path.endswith(".pkl"):
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported model format for {model_path}")
        
        logging.info(f"Model saved to {model_path}")
        return model_path
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific run."""
        artifact_path = os.path.join(
            self.config.get("folder_path", {}).get("artifacts", "artifacts"),
            run_id,
            "run_context.json"
        )
        
        if os.path.exists(artifact_path):
            with open(artifact_path, "r") as f:
                return json.load(f)
        else:
            logging.warning(f"Run details not found for {run_id}")
            return None
    
    def list_runs(self, limit: int = 10, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent runs with optional filtering by status."""
        base_dir = self.config.get("folder_path", {}).get("artifacts", "artifacts")
        if not os.path.exists(base_dir):
            return []
        
        runs = []
        for run_id in os.listdir(base_dir):
            run_context_path = os.path.join(base_dir, run_id, "run_context.json")
            if os.path.exists(run_context_path):
                with open(run_context_path, "r") as f:
                    run_context = json.load(f)
                    
                    if status is None or run_context.get("status") == status:
                        runs.append(run_context)
        
        # Sort by start time descending and limit results
        runs.sort(key=lambda r: r.get("start_time", ""), reverse=True)
        return runs[:limit]


# -----------------------------
# Model Registry
# -----------------------------

class ModelRegistry:
    """A registry for tracking model versions and their metadata."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        base_dir = self.config.get("folder_path", {}).get("models", "models")
        self.registry_path = os.path.join(base_dir, "registry.json")
        self.models_path = base_dir
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({}, f)
    
    def register_model(self, model: Any, name: str, version: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model in the registry and return its version."""
        # Load current registry
        with open(self.registry_path, "r") as f:
            registry = json.load(f)
        
        # Initialize model entry if it doesn't exist
        if name not in registry:
            registry[name] = {"versions": {}}
        
        # Generate version if not provided
        if version is None:
            existing_versions = registry[name]["versions"].keys()
            if existing_versions:
                latest_version = max(int(v) for v in existing_versions if v.isdigit())
                version = str(latest_version + 1)
            else:
                version = "1"
        
        # Create model directory
        model_dir = os.path.join(self.models_path, name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model file
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Update registry with metadata
        entry_metadata = metadata or {}
        entry_metadata["created_at"] = datetime.now().isoformat()
        
        registry[name]["versions"][version] = {
            "path": model_path,
            "metadata": entry_metadata
        }
        
        # Save registry
        with open(self.registry_path, "w") as f:
            json.dump(registry, f)
        
        logging.info(f"Model '{name}' version '{version}' registered at {model_path}")
        return version
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """Load a model from the registry."""
        # Load registry
        with open(self.registry_path, "r") as f:
            registry = json.load(f)
        
        if name not in registry:
            raise ValueError(f"Model '{name}' not found in registry")
        
        # Determine version to load
        versions = registry[name]["versions"]
        if not versions:
            raise ValueError(f"No versions found for model '{name}'")
        
        if version is None:
            # Find latest numeric version
            numeric_versions = [v for v in versions.keys() if v.isdigit()]
            if numeric_versions:
                version = str(max(int(v) for v in numeric_versions))
            else:
                version = list(versions.keys())[0]
        
        if version not in versions:
            raise ValueError(f"Version '{version}' not found for model '{name}'")
        
        # Load model
        model_path = versions[version]["path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        logging.info(f"Loaded model '{name}' version '{version}' from {model_path}")
        return model
    
    def get_model_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific model version."""
        # Load registry
        with open(self.registry_path, "r") as f:
            registry = json.load(f)
        
        if name not in registry:
            raise ValueError(f"Model '{name}' not found in registry")
        
        versions = registry[name]["versions"]
        if not versions:
            raise ValueError(f"No versions found for model '{name}'")
        
        if version is None:
            # Find latest numeric version
            numeric_versions = [v for v in versions.keys() if v.isdigit()]
            if numeric_versions:
                version = str(max(int(v) for v in numeric_versions))
            else:
                version = list(versions.keys())[0]
        
        if version not in versions:
            raise ValueError(f"Version '{version}' not found for model '{name}'")
        
        return versions[version]["metadata"]
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions in the registry."""
        # Load registry
        with open(self.registry_path, "r") as f:
            registry = json.load(f)
        
        result = {}
        for name, details in registry.items():
            result[name] = list(details["versions"].keys())
        
        return result


# ----------------------------- 
# Example Usage
# ----------------------------- 

def create_example_ml_pipeline():
    """Create an example ML pipeline for demonstration."""
    pipeline = Pipeline(name="ml_example", description="An example ML pipeline")
    

    def load_data(context):
        """Load and prepare dataset."""
        logging.info("Loading dataset")
        
        # Simulate loading data
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Log data statistics
        context["metrics_store"].log_metric("data_size", len(y))
        context["metrics_store"].log_metric("features_count", X.shape[1])
        
        # Store data as artifact
        context["artifact_store"].save_artifact(
            {"X": X, "y": y}, "data", "dataset.pkl"
        )
        
        return {"X": X, "y": y}
    

    def train_model(context):
        """Train a model on the data."""
        logging.info("Training model")
        
        # Get data from previous task
        data = context["results"]["load_data"]
        X, y = data["X"], data["y"]
        
        # Simulate training a model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Log model metrics
        import numpy as np
        train_preds = model.predict(X)
        accuracy = np.mean(train_preds == y)
        context["metrics_store"].log_metric("train_accuracy", accuracy)
        
        # Store model as artifact
        context["artifact_store"].save_artifact(
            model, "models", "model.pkl"
        )
        
        return model
    

    def evaluate_model(context):
        """Evaluate the trained model."""
        logging.info("Evaluating model")
        
        # Get data and model
        data = context["results"]["load_data"]
        model = context["results"]["train_model"]
        X, y = data["X"], data["y"]
        
        # Simulate evaluation (using training data for simplicity)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        predictions = model.predict(X)
        
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0)
        }
        
        # Log metrics
        for name, value in metrics.items():
            context["metrics_store"].log_metric(name, value)
        
        # Store evaluation results as artifact
        context["artifact_store"].save_artifact(
            metrics, "evaluation", "metrics.json"
        )
        
        return metrics
    
    # Add tasks to pipeline
    load_data_idx = pipeline.add_task(load_data)
    train_model_idx = pipeline.add_task(train_model)
    evaluate_model_idx = pipeline.add_task(evaluate_model)
    
    # Set up dependencies
    pipeline.add_dependency(train_model_idx, load_data_idx)
    pipeline.add_dependency(evaluate_model_idx, train_model_idx)
    
    return pipeline


def run_example():
    """Run the example ML pipeline."""
    # Create config
    config = {
        "folder_path": {
            "artifacts": "artifacts",
            "metrics": "metrics",
            "models": "models"
        }
    }
    
    # Create stack
    ml_stack = Stack(name="ml_pipeline", config=config)
    
    # Create model registry
    model_registry = ModelRegistry(ml_stack.config)
    
    # Create and run pipeline
    pipeline = create_example_ml_pipeline()
    run_id = pipeline.run(ml_stack, tags={"env": "dev", "purpose": "demo"})
    
    # List artifacts
    artifacts = ml_stack.artifact_store.list_artifacts(run_id)
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
        # Load the trained model
        model_path = os.path.join("artifacts", run_id, "models", "model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load evaluation metrics
        metrics_path = os.path.join("artifacts", run_id, "evaluation", "metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Register model with metadata
        version = model_registry.register_model(
            model=model,
            name="example_model",
            metadata={
                "run_id": run_id,
                "metrics": metrics,
                "tags": run_details["tags"]
            }
        )
        
        print(f"\nModel registered as 'example_model' version '{version}'")
    
    return run_id, ml_stack, model_registry


if __name__ == "__main__":
    run_example()