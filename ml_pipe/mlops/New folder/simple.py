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


class MetadataTracker:
    """Tracks and stores metadata about pipeline runs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.metadata_store = None
        
    def initialize(self):
        """Initialize the appropriate metadata store based on configuration."""
        if self.config.get("metadata_store.type") == "mlflow":
            from integrations.mlflow.mlflow_tracking import MLflowTracking
            self.metadata_store = MLflowTracking(self.config)
            
    def capture_metadata(self, metadata: PipelineMetadata):
        """Capture and store pipeline metadata."""
        if self.metadata_store:
            self.metadata_store.start_run(run_name=metadata.run_id)
            self.metadata_store.log_params(metadata.parameters)
            self.metadata_store.log_metrics(metadata.metrics)
            for artifact_name, artifact_path in metadata.artifacts.items():
                self.metadata_store.log_artifact(artifact_path)
            self.metadata_store.end_run()
            
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get run information by run ID."""
        if self.metadata_store:
            return self.metadata_store.get_run(run_id)
        return None





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
        
        if metadata_store:
            metadata_store.start_run(run_name=f"{self.name}-{run_id}", tags=tags)
        
        start_time = datetime.now()
        status = "RUNNING"
        
        try:
            # Execute each step
            for step in self.steps:
                output = step.execute(context, artifact_store)
                artifact_id = artifact_store.store(step.name, output)
                context[step.name] = artifact_id
                
                if metadata_store:
                    metadata_store.log_artifact(step.name, artifact_id)
            
            status = "COMPLETED"
        except Exception as e:
            status = "FAILED"
            if metadata_store:
                metadata_store.log_param("error", str(e))
            raise
        finally:
            end_time = datetime.now()
            if metadata_store:
                metadata_store.end_run()
            
            # Create run object
            run = Run(
                run_id=run_id,
                status=status,
                start_time=start_time,
                end_time=end_time
            )
            
            # Return the run ID
            return run_id





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

    def set_metadata_store(self, metadata_store):
        """Set the metadata store for this stack."""
        self.metadata_store = metadata_store


# ----------------------------- 
# Example Usage
# ----------------------------- 

def create_example_pipeline():
    """Create an example pipeline for demonstration."""
    

    def my_pipeline(pipe):

        def first_step():
            """Generate initial data."""
            return "Initial data"
        

        def second_step(first_step):
            """Process the data from the first step."""
            return f"Processed: {first_step}"
        

        def third_step(second_step):
            """Further process the data from the second step."""
            return f"Further processed: {second_step}"
    
    return my_pipeline


def run_example():
    """Run the example pipeline."""
    # Create config
    config = {
        "folder_path": {
            "artifacts": "artifacts",
            "metadata": "metadata"
        }
    }
    
    # Create stack
    dev_stack = Stack(name="ml_pipeline", config=config)
    
    # Create and run pipeline
    pipeline = create_example_pipeline()
    run_id = pipeline.run(dev_stack, tags={"env": "dev"})
    
    # Get run metadata
    run = dev_stack.metadata_store.get_run(run_id) if dev_stack.metadata_store else None
    
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
    
    return run_id, dev_stack


if __name__ == "__main__":
    run_example()
