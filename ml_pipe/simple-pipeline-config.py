import os
import uuid
import yaml
import json
import pickle
import logging
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

    def get_version_info(self) -> Dict[str, str]:
        """Return version information for the current run."""
        return {
            "run_id": self.run_id,
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
# Pipeline
# -----------------------------

class Step:
    """A single step in a pipeline."""
    
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        
    def execute(self, context, artifact_store):
        """Execute the step function with the given context."""
        # Get function parameters and match with context
        return self.func()


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
        versioner = stack.versioner
        
        # Get run ID
        run_id = versioner.get_version_info()["run_id"]
        
        start_time = datetime.now()
        status = "RUNNING"
        
        try:
            # Execute each step
            for step in self.steps:
                output = step.execute(context, artifact_store)
                artifact_id = artifact_store.store(step.name, output)
                context[step.name] = artifact_id
            
            status = "COMPLETED"
        except Exception as e:
            status = "FAILED"
            raise
        finally:
            end_time = datetime.now()
            
            # Return the run ID
            return run_id


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


# ----------------------------- 
# Example Usage
# ----------------------------- 

def create_example_pipeline():
    """Create an example pipeline for demonstration."""
    pipeline = Pipeline(name="example_pipeline", description="A simple example pipeline")
    
    @pipeline.step
    def first_step():
        """Generate initial data."""
        return "Initial data"
    
    @pipeline.step
    def second_step():
        """Process the data from the first step."""
        return "Processed data"
    
    @pipeline.step
    def third_step():
        """Further process the data from the second step."""
        return "Final processed data"
    
    return pipeline


def run_example():
    """Run the example pipeline."""
    # Create config
    config = {
        "folder_path": {
            "artifacts": "artifacts"
        }
    }
    
    # Create stack
    dev_stack = Stack(name="simple_pipeline", config=config)
    
    # Create and run pipeline
    pipeline = create_example_pipeline()
    run_id = pipeline.run(dev_stack, tags={"env": "dev"})
    
    # List artifacts
    artifacts = dev_stack.artifact_store.list_artifacts(run_id)
    print(f"Run ID: {run_id}")
    print("Artifacts:")
    for uri in artifacts:
        print(f"- {uri}")
    
    return run_id, dev_stack


if __name__ == "__main__":
    run_example()
