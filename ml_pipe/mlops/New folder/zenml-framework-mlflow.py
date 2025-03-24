import os
import uuid
import yaml
import json
import pickle
import inspect
import logging
import subprocess
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional


# -----------------------------
# Config
# -----------------------------

class Config:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with optional default."""
        keys = key.split(".")
        result = self.config_dict
        for k in keys:
            if isinstance(result, dict) and k in result:
                result = result[k]
            else:
                return default
        return result

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
# Versioning
# -----------------------------

class PipelineVersioner:
    """Handles versioning for pipeline runs."""
    
    def __init__(self):
        self.run_id = str(uuid.uuid4())
        self.git_commit = self._get_git_commit()

    def _get_git_commit(self) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("utf-8").strip()
        except Exception:
            return None

    def get_version_info(self) -> Dict[str, str]:
        """Return version information for the current run."""
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "version_date": datetime.now().isoformat()
        }


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
        base_dir = self.config.get("folder_path.artifacts", "artifacts")
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
        elif name.endswith(".csv"):
            if isinstance(artifact, pd.DataFrame):
                artifact.to_csv(artifact_path, index=False)
            else:
                raise ValueError("CSV format only supports pandas DataFrames.")
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
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
            else:
                raise ValueError(f"Unsupported format for {name}")
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
            self.config.get("folder_path.artifacts", "artifacts"),
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
# Metadata
# -----------------------------

@dataclass
class PipelineMetadata:
    """Data class to store metadata about a pipeline run."""
    run_id: str
    execution_date: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    git_commit: str


@dataclass
class Run:
    """Data class representing a pipeline run."""
    run_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None


class MetadataStore:
    """Base class for metadata storage backends."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new run."""
        raise NotImplementedError("Subclasses must implement start_run")
    
    def end_run(self) -> None:
        """End the current run."""
        raise NotImplementedError("Subclasses must implement end_run")
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        raise NotImplementedError("Subclasses must implement log_param")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float) -> None:
        """Log a metric."""
        raise NotImplementedError("Subclasses must implement log_metric")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value)
    
    def log_artifact(self, key: str, artifact_path: str) -> None:
        """Log an artifact."""
        raise NotImplementedError("Subclasses must implement log_artifact")
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        raise NotImplementedError("Subclasses must implement get_run")


# -----------------------------
# MLflow Integration 
# -----------------------------

class MLflowTracking(MetadataStore):
    """MLflow implementation of metadata storage."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._setup_mlflow()
        self.active_run = None
        self.active_run_id = None
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking server connection."""
        try:
            import mlflow
            
            # Configure MLflow tracking URI
            tracking_uri = self.config.get("mlflow.tracking_uri", "")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logging.info(f"MLflow tracking URI set to {tracking_uri}")
            
            # Configure experiment
            experiment_name = self.config.get("mlflow.experiment_name", "default")
            mlflow.set_experiment(experiment_name)
            logging.info(f"MLflow experiment set to {experiment_name}")
            
            self.mlflow = mlflow
        except ImportError:
            logging.error("MLflow not installed. Please install it with 'pip install mlflow'")
            raise
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run."""
        tags = tags or {}
        self.active_run = self.mlflow.start_run(run_name=run_name, tags=tags)
        self.active_run_id = self.active_run.info.run_id
        logging.info(f"MLflow run started with ID: {self.active_run_id}")
        return self.active_run_id
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.active_run:
            self.mlflow.end_run()
            logging.info(f"MLflow run {self.active_run_id} ended")
            self.active_run = None
            self.active_run_id = None
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow."""
        self.mlflow.log_param(key, value)
    
    def log_metric(self, key: str, value: float) -> None:
        """Log a metric to MLflow."""
        self.mlflow.log_metric(key, value)
    
    def log_artifact(self, key: str, artifact_path: str) -> None:
        """Log an artifact to MLflow."""
        if os.path.exists(artifact_path):
            self.mlflow.log_artifact(artifact_path)
        else:
            logging.warning(f"Cannot log artifact {key} - path {artifact_path} does not exist")
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID from MLflow."""
        try:
            mlflow_run = self.mlflow.get_run(run_id)
            start_time = datetime.fromtimestamp(mlflow_run.info.start_time / 1000.0)
            end_time = None
            if mlflow_run.info.end_time:
                end_time = datetime.fromtimestamp(mlflow_run.info.end_time / 1000.0)
            
            return Run(
                run_id=run_id,
                status=mlflow_run.info.status,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            logging.error(f"Error getting MLflow run {run_id}: {e}")
            return None


class MetadataTracker:
    """Tracks and stores metadata about pipeline runs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.metadata_store = None
        
    def initialize(self):
        """Initialize the appropriate metadata store based on configuration."""
        metadata_store_type = self.config.get("metadata_store.type")
        
        if metadata_store_type == "mlflow":
            self.metadata_store = MLflowTracking(self.config)
            logging.info("MLflow metadata store initialized")
        else:
            logging.warning(f"Unsupported metadata store type: {metadata_store_type}")
            
    def capture_metadata(self, metadata: PipelineMetadata):
        """Capture and store pipeline metadata."""
        if self.metadata_store:
            self.metadata_store.start_run(run_name=metadata.run_id)
            self.metadata_store.log_params(metadata.parameters)
            self.metadata_store.log_metrics(metadata.metrics)
            for artifact_name, artifact_path in metadata.artifacts.items():
                self.metadata_store.log_artifact(artifact_name, artifact_path)
            self.metadata_store.end_run()
            
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get run information by run ID."""
        if self.metadata_store:
            return self.metadata_store.get_run(run_id)
        return None


# -----------------------------
# Step
# -----------------------------

class Step:
    """A single step in a pipeline."""
    
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        self.description = func.__doc__ or ""

    def execute(self, context: Dict[str, Any], artifact_store: ArtifactStore):
        """Execute the step function with the appropriate context."""
        logging.info(f"Executing step: {self.name}")
        sig = inspect.signature(self.func)
        kwargs = {
            param.name: artifact_store.get_artifact(context[param.name]) 
            for param in sig.parameters.values() 
            if param.name in context
        }
        return self.func(**kwargs)


# -----------------------------
# Pipeline
# -----------------------------

class Pipeline:
    """A pipeline composed of steps."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[Step] = []

    def step(self, func=None, *, name=None):
        """Decorator to add a step to the pipeline."""
        if func is None:
            return lambda f: self._add_step(f, name)
        return self._add_step(func, name)

    def _add_step(self, func, name=None):
        """Add a step to the pipeline."""
        step_instance = Step(func, name=name)
        self.steps.append(step_instance)
        return func

    def run(self, stack: "Stack", tags: Optional[Dict[str, str]] = None) -> str:
        """Run the pipeline with the given stack."""
        tags = tags or {}
        context = {}
        artifact_store = stack.artifact_store
        metadata_store = stack.metadata_store
        versioner = stack.versioner
        
        # Get run ID and start tracking
        run_id = versioner.get_version_info()["run_id"]
        
        # Add pipeline info to tags
        tags["pipeline.name"] = self.name
        tags["pipeline.description"] = self.description
        tags["git.commit"] = versioner.get_version_info().get("git_commit", "unknown")
        
        if metadata_store:
            metadata_store.start_run(run_name=f"{self.name}-{run_id}", tags=tags)
            # Log pipeline metadata
            metadata_store.log_param("pipeline.name", self.name)
            metadata_store.log_param("pipeline.description", self.description)
            metadata_store.log_param("pipeline.steps", len(self.steps))
        
        start_time = datetime.now()
        status = "RUNNING"
        artifacts = {}
        parameters = {}
        metrics = {}
        
        try:
            # Execute each step
            for step in self.steps:
                step_start_time = datetime.now()
                output = step.execute(context, artifact_store)
                step_end_time = datetime.now()
                artifact_id = artifact_store.store(step.name, output)
                context[step.name] = artifact_id
                
                # Save output as artifact if possible
                try:
                    artifact_path = artifact_store.save_artifact(
                        output, 
                        "outputs", 
                        f"{step.name}.pkl"
                    )
                    artifacts[step.name] = artifact_path
                    
                    if metadata_store:
                        metadata_store.log_artifact(step.name, artifact_path)
                except Exception as e:
                    logging.warning(f"Could not save artifact for step {step.name}: {e}")
                
                # Log step execution time
                step_duration = (step_end_time - step_start_time).total_seconds()
                metrics[f"step.{step.name}.duration"] = step_duration
                
                if metadata_store:
                    metadata_store.log_metric(f"step.{step.name}.duration", step_duration)
            
            status = "COMPLETED"
        except Exception as e:
            status = "FAILED"
            logging.error(f"Pipeline execution failed: {e}")
            parameters["error"] = str(e)
            
            if metadata_store:
                metadata_store.log_param("error", str(e))
            raise
        finally:
            end_time = datetime.now()
            
            # Calculate overall duration
            duration = (end_time - start_time).total_seconds()
            metrics["pipeline.duration"] = duration
            
            if metadata_store:
                metadata_store.log_metric("pipeline.duration", duration)
                metadata_store.log_param("pipeline.status", status)
                metadata_store.end_run()
            
            # Create and store pipeline metadata
            metadata = PipelineMetadata(
                run_id=run_id,
                execution_date=start_time,
                parameters=parameters,
                metrics=metrics,
                artifacts=artifacts,
                git_commit=versioner.get_version_info().get("git_commit", "unknown")
            )
            
            stack.metadata_tracker.capture_metadata(metadata)
            
            # Return the run ID
            return run_id


# -----------------------------
# Decorators
# -----------------------------

def step(func=None, *, name=None):
    """Decorator to create a step."""
    if func is None:
        return lambda f: Step(f, name=name)
    return Step(func, name=name)


def pipeline(name: str, description: str = ""):
    """Decorator to create a pipeline."""
    def decorator(func):
        pipe = Pipeline(name, description=description)
        func(pipe)
        return pipe
    return decorator


# -----------------------------
# Orchestration
# -----------------------------

class Orchestrator:
    """Base class for orchestration backends."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def schedule(self, pipeline: Pipeline, schedule: str):
        """Schedule a pipeline to run according to the given schedule."""
        raise NotImplementedError("Subclasses must implement schedule method")


# -----------------------------
# Stack
# -----------------------------

class Stack:
    """A stack that brings together all components needed to run pipelines."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = Config(config or {})
        self.versioner = PipelineVersioner()
        
        # Initialize components with versioning
        self.artifact_store = ArtifactStore(
            self.config, 
            self.versioner.get_version_info()["run_id"]
        )
        
        self.metadata_tracker = MetadataTracker(self.config)
        self.metadata_tracker.initialize()
        self.metadata_store = self.metadata_tracker.metadata_store
        self.orchestrator = None

    def set_orchestrator(self, orchestrator: Orchestrator):
        """Set the orchestrator for this stack."""
        self.orchestrator = orchestrator

    def set_metadata_store(self, metadata_store: MetadataStore):
        """Set the metadata store for this stack."""
        self.metadata_store = metadata_store


# ----------------------------- 
# Example Usage with MLflow
# ----------------------------- 

def create_example_pipeline():
    """Create an example pipeline for demonstration."""
    
    @pipeline("example_pipeline", description="An example pipeline with MLflow tracking")
    def my_pipeline(pipe):
        @pipe.step
        def load_data():
            """Load data for processing."""
            return pd.DataFrame({
                'id': range(1, 11),
                'value': [i * 2 for i in range(1, 11)]
            })
        
        @pipe.step
        def preprocess(load_data):
            """Preprocess the data."""
            df = load_data.copy()
            df['normalized'] = df['value'] / df['value'].max()
            return df
        
        @pipe.step
        def calculate_metrics(preprocess):
            """Calculate metrics on the data."""
            df = preprocess
            metrics = {
                'mean': df['value'].mean(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max()
            }
            return metrics
    
    return my_pipeline


def run_mlflow_example():
    """Run the example pipeline with MLflow tracking."""
    # Create config with MLflow settings
    config = {
        "folder_path": {
            "artifacts": "artifacts",
            "metadata": "metadata"
        },
        "metadata_store": {
            "type": "mlflow"
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",  # Default MLflow server port
            "experiment_name": "zenml_example"
        }
    }
    
    # Create stack
    dev_stack = Stack(name="dev", config=config)
    
    # Create and run pipeline
    pipeline = create_example_pipeline()
    run_id = pipeline.run(dev_stack, tags={"env": "dev"})
    
    print(f"Pipeline run completed with run_id: {run_id}")
    
    # Get run metadata if MLflow is available
    if dev_stack.metadata_store:
        run = dev_stack.metadata_store.get_run(run_id)
        if run:
            print(f"Run ID: {run.run_id}")
            print(f"Status: {run.status}")
            print(f"Start time: {run.start_time}")
            print(f"End time: {run.end_time}")
    
    # List artifacts
    artifacts = dev_stack.artifact_store.list_artifacts(run_id)
    print("Artifacts:")
    for uri in artifacts:
        print(f"- {uri}")
    
    print(f"\nYou can view this run in the MLflow UI at: http://localhost:5000")
    
    return run_id, dev_stack


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    try:
        run_mlflow_example()
    except ImportError as e:
        logging.error(f"Error: {e}")
        logging.error("MLflow integration requires MLflow to be installed.")
        logging.error("Install it with: pip install mlflow")
