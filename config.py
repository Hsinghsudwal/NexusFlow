# custom_mlops/core.py
"""Core module for the custom MLOps framework."""

import os
import yaml
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLOpsConfig:
    """Configuration manager for the MLOps framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.getcwd(), "mlops_config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return {}
        else:
            logger.info(f"Config file not found at {self.config_path}. Using default config.")
            return {}
    
    def save_config(self):
        """Save current configuration to YAML file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f)
            logger.info(f"Config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(config_dict)


class Artifact:
    """Represents a data artifact in the ML pipeline."""
    
    def __init__(self, name: str, data: Any, metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now().isoformat()
    
    def save(self, artifact_store: 'ArtifactStore'):
        """Save the artifact to a store."""
        return artifact_store.save_artifact(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class ArtifactStore:
    """Base class for storing and retrieving artifacts."""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.join(os.getcwd(), "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
    
    def save_artifact(self, artifact: Artifact) -> str:
        """Save an artifact and return its path."""
        artifact_dir = os.path.join(self.base_path, artifact.id)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(artifact_dir, "metadata.json"), 'w') as f:
            json.dump(artifact.to_dict(), f)
        
        # This is a simple implementation - in practice, different serialization
        # methods would be used based on the artifact type
        try:
            import pickle
            with open(os.path.join(artifact_dir, "data.pkl"), 'wb') as f:
                pickle.dump(artifact.data, f)
        except Exception as e:
            logger.error(f"Error saving artifact data: {e}")
            raise
        
        return artifact_dir
    
    def load_artifact(self, artifact_id: str) -> Artifact:
        """Load an artifact by ID."""
        artifact_dir = os.path.join(self.base_path, artifact_id)
        
        if not os.path.exists(artifact_dir):
            raise FileNotFoundError(f"Artifact {artifact_id} not found")
        
        # Load metadata
        with open(os.path.join(artifact_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Load data
        try:
            import pickle
            with open(os.path.join(artifact_dir, "data.pkl"), 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading artifact data: {e}")
            raise
        
        artifact = Artifact(metadata["name"], data, metadata["metadata"])
        artifact.id = metadata["id"]
        artifact.created_at = metadata["created_at"]
        
        return artifact
    
    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List all artifacts in the store."""
        artifacts = []
        for artifact_id in os.listdir(self.base_path):
            try:
                metadata_path = os.path.join(self.base_path, artifact_id, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        artifacts.append(json.load(f))
            except Exception as e:
                logger.error(f"Error loading artifact metadata: {e}")
        
        return artifacts


class Step:
    """Base class for pipeline steps."""
    
    def __init__(self, name: str = None):
        self.id = str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.inputs = {}
        self.outputs = {}
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the step and return outputs."""
        raise NotImplementedError("Subclasses must implement run()")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.__class__.__name__
        }


class Pipeline:
    """Manages the execution of a sequence of steps."""
    
    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.steps = []
        self.artifact_store = None
        self.metadata = {}
        self.created_at = datetime.datetime.now().isoformat()
    
    def add_step(self, step: Step) -> 'Pipeline':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def set_artifact_store(self, artifact_store: ArtifactStore) -> 'Pipeline':
        """Set the artifact store for the pipeline."""
        self.artifact_store = artifact_store
        return self
    
    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the pipeline steps in sequence."""
        if not self.artifact_store:
            self.artifact_store = ArtifactStore()
        
        input_data = input_data or {}
        step_outputs = {}
        pipeline_output = {}
        
        try:
            for step in self.steps:
                logger.info(f"Running step: {step.name}")
                
                # Prepare step inputs
                step_inputs = {**input_data, **step_outputs}
                
                # Run the step
                step_result = step.run(**step_inputs)
                
                # Store artifacts
                for key, value in step_result.items():
                    artifact = Artifact(f"{step.name}_{key}", value)
                    artifact.save(self.artifact_store)
                    pipeline_output[f"{step.name}.{key}"] = artifact.id
                
                # Update step outputs for next steps
                step_outputs.update(step_result)
            
            # Save pipeline run metadata
            run_metadata = {
                "pipeline_id": self.id,
                "name": self.name,
                "status": "completed",
                "start_time": self.created_at,
                "end_time": datetime.datetime.now().isoformat(),
                "outputs": pipeline_output
            }
            
            run_id = str(uuid.uuid4())
            run_dir = os.path.join(os.getcwd(), "pipeline_runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
                json.dump(run_metadata, f)
            
            logger.info(f"Pipeline run completed. Run ID: {run_id}")
            return step_outputs
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    def save(self, path: str = None) -> str:
        """Save pipeline definition to disk."""
        path = path or os.path.join(os.getcwd(), "pipelines", f"{self.name}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Pipeline saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        """Load pipeline definition from disk."""
        with open(path, 'r') as f:
            pipeline_dict = json.load(f)
        
        pipeline = cls(pipeline_dict["name"])
        pipeline.id = pipeline_dict["id"]
        pipeline.metadata = pipeline_dict["metadata"]
        pipeline.created_at = pipeline_dict["created_at"]
        
        # Note: This simplified loader doesn't recreate steps
        # A full implementation would need to map step types to classes
        
        return pipeline


class StepDecorator:
    """Decorator to convert functions into pipeline steps."""
    
    @staticmethod
    def step(name: str = None):
        def decorator(func: Callable):
            class FunctionStep(Step):
                def __init__(self):
                    super().__init__(name or func.__name__)
                    self.func = func
                
                def run(self, **kwargs):
                    return {"output": self.func(**kwargs)}
            
            return FunctionStep()
        
        return decorator


# Export common decorators
step = StepDecorator.step