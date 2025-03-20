import os
import json
import yaml
import pickle
import logging
import inspect
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Union, Type
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mlops_framework")

# ----------------------------- 
# Config Management
# ----------------------------- 
class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}
    
    def get(self, key: str, default: Any = None):
        """Get a configuration value by key with an optional default."""
        return self.config_dict.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config_dict[key] = value
        return self
    
    def update(self, config_dict: Dict):
        """Update multiple configuration values at once."""
        self.config_dict.update(config_dict)
        return self
    
    def to_dict(self):
        """Return the configuration as a dictionary."""
        return self.config_dict.copy()
    
    def save(self, config_path: str):
        """Save configuration to a file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            with open(config_path, "w") as file:
                if config_path.endswith((".yml", ".yaml")):
                    yaml.dump(self.config_dict, file, default_flow_style=False)
                else:
                    json.dump(self.config_dict, file, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            raise ValueError(f"Error saving config file {config_path}: {e}")
    
    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    import yaml
                    config_data = yaml.safe_load(file)
                else:
                    import json
                    config_data = json.load(file)
                return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")

# ----------------------------- 
# Metadata Store
# ----------------------------- 
@dataclass
class RunMetadata:
    run_id: str
    pipeline_name: str
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    end_time: Optional[datetime.datetime] = None
    status: str = "STARTED"  # STARTED, RUNNING, COMPLETED, FAILED
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    step_metadata: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary with datetime objects converted to ISO format."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary, parsing ISO format datetime strings."""
        if data.get('start_time'):
            data['start_time'] = datetime.datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.datetime.fromisoformat(data['end_time'])
        return cls(**data)

class MetadataStore:
    """Stores and retrieves metadata about pipeline runs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_path = config.get("folder_path", {}).get("metadata", "metadata")
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Metadata store initialized at '{self.base_path}'")
        self._active_runs = {}  # In-memory cache of active runs
    
    def create_run(self, pipeline_name: str, parameters: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str:
        """Create a new run and return its ID."""
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{pipeline_name}"
        run = RunMetadata(
            run_id=run_id,
            pipeline_name=pipeline_name,
            parameters=parameters or {},
            tags=tags or {}
        )
        self._active_runs[run_id] = run
        self._save_run_metadata(run)
        return run_id
    
    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Get metadata for a specific run."""
        if run_id in self._active_runs:
            return self._active_runs[run_id]
        
        run_path = os.path.join(self.base_path, f"{run_id}.json")
        if os.path.exists(run_path):
            with open(run_path, 'r') as f:
                run_data = json.load(f)
            return RunMetadata.from_dict(run_data)
        
        logger.warning(f"Run {run_id} not found")
        return None
    
    def update_run_status(self, run_id: str, status: str, end_time: bool = False) -> bool:
        """Update the status of a run."""
        run = self.get_run(run_id)
        if not run:
            return False
            
        run.status = status
        if end_time:
            run.end_time = datetime.datetime.now()
        
        self._active_runs[run_id] = run
        self._save_run_metadata(run)
        return True
    
    def log_metric(self, run_id: str, key: str, value: Any) -> bool:
        """Log a metric for a run."""
        run = self.get_run(run_id)
        if not run:
            return False
            
        run.metrics[key] = value
        self._active_runs[run_id] = run
        self._save_run_metadata(run)
        return True
    
    def log_step_start(self, run_id: str, step_name: str) -> bool:
        """Log the start of a step."""
        run = self.get_run(run_id)
        if not run:
            return False
            
        if step_name not in run.step_metadata:
            run.step_metadata[step_name] = {}
            
        run.step_metadata[step_name]["start_time"] = datetime.datetime.now().isoformat()
        run.step_metadata[step_name]["status"] = "RUNNING"
        
        self._active_runs[run_id] = run
        self._save_run_metadata(run)
        return True
    
    def log_step_end(self, run_id: str, step_name: str, status: str = "COMPLETED") -> bool:
        """Log the end of a step."""
        run = self.get_run(run_id)
        if not run:
            return False
            
        if step_name not in run.step_metadata:
            run.step_metadata[step_name] = {}
            
        run.step_metadata[step_name]["end_time"] = datetime.datetime.now().isoformat()
        run.step_metadata[step_name]["status"] = status
        
        self._active_runs[run_id] = run
        self._save_run_metadata(run)
        return True
    
    def list_runs(self, pipeline_name: Optional[str] = None, tags: Dict[str, str] = None) -> List[RunMetadata]:
        """List all runs, optionally filtered by pipeline name and tags."""
        runs = []
        
        # First, check in-memory cache
        for run in self._active_runs.values():
            if pipeline_name and run.pipeline_name != pipeline_name:
                continue
                
            if tags and not all(run.tags.get(k) == v for k, v in tags.items()):
                continue
                
            runs.append(run)
        
        # Then check saved metadata files
        for filename in os.listdir(self.base_path):
            if not filename.endswith('.json'):
                continue
                
            run_id = filename[:-5]  # Remove '.json'
            if run_id in self._active_runs:
                continue  # Already added from cache
                
            try:
                with open(os.path.join(self.base_path, filename), 'r') as f:
                    run_data = json.load(f)
                run = RunMetadata.from_dict(run_data)
                
                if pipeline_name and run.pipeline_name != pipeline_name:
                    continue
                    
                if tags and not all(run.tags.get(k) == v for k, v in tags.items()):
                    continue
                    
                runs.append(run)
            except Exception as e:
                logger.error(f"Error loading run metadata from {filename}: {e}")
        
        return runs
    
    def _save_run_metadata(self, run: RunMetadata):
        """Save run metadata to disk."""
        run_path = os.path.join(self.base_path, f"{run.run_id}.json")
        try:
            with open(run_path, 'w') as f:
                json.dump(run.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving run metadata for {run.run_id}: {e}")

# ----------------------------- 
# Artifact Store
# ----------------------------- 
class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_path = config.get("folder_path", {}).get("artifacts", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        self._cache = {}  # In-memory cache
        logger.info(f"Artifact store initialized at '{self.base_path}'")
    
    def save_artifact(
        self, 
        artifact: Any, 
        run_id: str, 
        step_name: str, 
        name: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Save an artifact and return its URI."""
        artifact_dir = os.path.join(self.base_path, run_id, step_name)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Determine file extension and save method based on artifact type
        if name.endswith(('.pkl', '.pickle')):
            file_path = os.path.join(artifact_dir, name)
            with open(file_path, "wb") as f:
                pickle.dump(artifact, f)
        elif name.endswith('.csv'):
            file_path = os.path.join(artifact_dir, name)
            if isinstance(artifact, pd.DataFrame):
                artifact.to_csv(file_path, index=False)
            else:
                raise ValueError("CSV format only supports pandas DataFrames")
        elif name.endswith('.json'):
            file_path = os.path.join(artifact_dir, name)
            with open(file_path, "w") as f:
                json.dump(artifact, f, indent=2)
        elif name.endswith(('.yml', '.yaml')):
            file_path = os.path.join(artifact_dir, name)
            with open(file_path, "w") as f:
                yaml.dump(artifact, f, default_flow_style=False)
        else:
            # Default to pickle for unknown types
            file_path = os.path.join(artifact_dir, f"{name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(artifact, f)
        
        # Generate URI
        uri = f"{run_id}/{step_name}/{os.path.basename(file_path)}"
        
        # Store in cache
        self._cache[uri] = artifact
        
        # Save metadata if provided
        if metadata:
            metadata_path = f"{file_path}.meta.json"
            with open(metadata_path, "w") as f:
                meta_with_type = metadata.copy()
                meta_with_type['_type'] = type(artifact).__name__
                meta_with_type['_created_at'] = datetime.datetime.now().isoformat()
                json.dump(meta_with_type, f, indent=2)
        
        logger.info(f"Artifact '{name}' saved to {file_path} (URI: {uri})")
        return uri
    
    def load_artifact(self, uri: str) -> Any:
        """Load an artifact by URI."""
        # Check cache first
        if uri in self._cache:
            return self._cache[uri]
        
        # Parse URI to get file path
        parts = uri.split('/')
        if len(parts) < 3:
            raise ValueError(f"Invalid artifact URI: {uri}")
        
        run_id = parts[0]
        step_name = parts[1]
        filename = parts[2]
        
        file_path = os.path.join(self.base_path, run_id, step_name, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Artifact not found at {file_path}")
            return None
        
        # Load based on file extension
        try:
            if file_path.endswith(('.pkl', '.pickle')):
                with open(file_path, "rb") as f:
                    artifact = pickle.load(f)
            elif file_path.endswith('.csv'):
                artifact = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, "r") as f:
                    artifact = json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                with open(file_path, "r") as f:
                    artifact = yaml.safe_load(f)
            else:
                # Default to pickle for unknown types
                with open(file_path, "rb") as f:
                    artifact = pickle.load(f)
            
            # Store in cache
            self._cache[uri] = artifact
            
            logger.info(f"Artifact loaded from {file_path} (URI: {uri})")
            return artifact
        except Exception as e:
            logger.error(f"Error loading artifact from {file_path}: {e}")
            return None
    
    def get_artifact_metadata(self, uri: str) -> Dict[str, Any]:
        """Get metadata for an artifact."""
        parts = uri.split('/')
        if len(parts) < 3:
            raise ValueError(f"Invalid artifact URI: {uri}")
        
        run_id = parts[0]
        step_name = parts[1]
        filename = parts[2]
        
        file_path = os.path.join(self.base_path, run_id, step_name, filename)
        metadata_path = f"{file_path}.meta.json"
        
        if not os.path.exists(metadata_path):
            return {}
        
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading artifact metadata from {metadata_path}: {e}")
            return {}
    
    def list_artifacts(self, run_id: str, step_name: Optional[str] = None) -> List[str]:
        """List all artifacts for a run, optionally filtered by step name."""
        run_dir = os.path.join(self.base_path, run_id)
        if not os.path.exists(run_dir):
            return []
        
        uris = []
        if step_name:
            step_dir = os.path.join(run_dir, step_name)
            if not os.path.exists(step_dir):
                return []
            
            for filename in os.listdir(step_dir):
                if filename.endswith('.meta.json'):
                    continue
                uris.append(f"{run_id}/{step_name}/{filename}")
        else:
            for step in os.listdir(run_dir):
                step_dir = os.path.join(run_dir, step)
                if not os.path.isdir(step_dir):
                    continue
                
                for filename in os.listdir(step_dir):
                    if filename.endswith('.meta.json'):
                        continue
                    uris.append(f"{run_id}/{step}/{filename}")
        
        return uris

# ----------------------------- 
# Step
# ----------------------------- 
class Step:
    """Represents a step in a pipeline."""
    
    def __init__(self, func: Callable, name: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = func.__doc__ or ""
        self.signature = inspect.signature(func)
    
    def __call__(self, *args, **kwargs):
        """Allow the step to be called directly."""
        return self.func(*args, **kwargs)
    
    def execute(self, context: Dict[str, Any], run_id: str, stack: "Stack") -> Any:
        """Execute the step within the pipeline context."""
        # Get dependencies from the context
        kwargs = {}
        for param_name in self.signature.parameters:
            if param_name in context:
                artifact_uri = context[param_name]
                kwargs[param_name] = stack.artifact_store.load_artifact(artifact_uri)
        
        # Log step start
        stack.metadata_store.log_step_start(run_id, self.name)
        
        try:
            # Execute the function
            logger.info(f"Executing step '{self.name}'")
            result = self.func(**kwargs)
            
            # Log step completion
            stack.metadata_store.log_step_end(run_id, self.name, status="COMPLETED")
            
            return result
        except Exception as e:
            # Log step failure
            stack.metadata_store.log_step_end(run_id, self.name, status="FAILED")
            logger.error(f"Step '{self.name}' failed: {e}")
            raise

# ----------------------------- 
# Pipeline
# ----------------------------- 
class Pipeline:
    """Represents a pipeline of steps."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[Step] = []
    
    def add_step(self, step: Step) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def step(self, func=None, *, name=None):
        """Decorator to add a step to the pipeline."""
        def decorator(f):
            step = Step(f, name=name)
            self.add_step(step)
            return f
        
        if func is None:
            return decorator
        return decorator(func)
    
    def run(self, stack: "Stack", parameters: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str:
        """Run the pipeline and return the run ID."""
        # Create a new run in the metadata store
        run_id = stack.metadata_store.create_run(
            pipeline_name=self.name,
            parameters=parameters,
            tags=tags
        )
        
        # Initialize context
        context = {}
        
        # Update run status
        stack.metadata_store.update_run_status(run_id, "RUNNING")
        
        try:
            # Execute each step
            for step in self.steps:
                result = step.execute(context, run_id, stack)
                
                # Store the result as an artifact
                if result is not None:
                    artifact_uri = stack.artifact_store.save_artifact(
                        artifact=result,
                        run_id=run_id,
                        step_name=step.name,
                        name=f"{step.name}_output",
                        metadata={"step": step.name}
                    )
                    
                    # Add to context for next steps
                    context[step.name] = artifact_uri
            
            # Mark run as completed
            stack.metadata_store.update_run_status(run_id, "COMPLETED", end_time=True)
            logger.info(f"Pipeline '{self.name}' completed successfully (Run ID: {run_id})")
            
        except Exception as e:
            # Mark run as failed
            stack.metadata_store.update_run_status(run_id, "FAILED", end_time=True)
            logger.error(f"Pipeline '{self.name}' failed: {e}")
            raise
        
        return run_id

# ----------------------------- 
# Stack
# ----------------------------- 
class Stack:
    """
    Represents a full MLOps stack with artifact store, metadata store,
    and potentially other components.
    """
    
    def __init__(
        self, 
        name: str, 
        config: Config = None, 
        artifact_store: ArtifactStore = None,
        metadata_store: MetadataStore = None
    ):
        self.name = name
        self.config = config or Config()
        
        # Initialize stores if not provided
        self.artifact_store = artifact_store or ArtifactStore(self.config)
        self.metadata_store = metadata_store or MetadataStore(self.config)
        
        logger.info(f"Stack '{name}' initialized")
    
    @contextmanager
    def run_context(self, pipeline: Pipeline, parameters: Dict[str, Any] = None, tags: Dict[str, str] = None):
        """Context manager for pipeline runs."""
        run_id = self.metadata_store.create_run(
            pipeline_name=pipeline.name,
            parameters=parameters,
            tags=tags
        )
        
        try:
            self.metadata_store.update_run_status(run_id, "RUNNING")
            yield run_id
            self.metadata_store.update_run_status(run_id, "COMPLETED", end_time=True)
        except Exception as e:
            self.metadata_store.update_run_status(run_id, "FAILED", end_time=True)
            logger.error(f"Pipeline '{pipeline.name}' failed: {e}")
            raise
    
    def register_pipeline(self, pipeline: Pipeline):
        """Register a pipeline with this stack."""
        logger.info(f"Pipeline '{pipeline.name}' registered with stack '{self.name}'")
        return pipeline

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
# Example Usage
# ----------------------------- 
def create_example_pipeline():
    @pipeline("example_pipeline", description="An example pipeline")
    def my_pipeline(pipe):
        @pipe.step
        def first_step():
            """Generate initial data."""
            return "Initial data"
        
        @pipe.step
        def second_step(first_step):
            """Process the data from the first step."""
            return f"Processed: {first_step}"
        
        @pipe.step
        def third_step(second_step):
            """Further process the data from the second step."""
            return f"Further processed: {second_step}"
        
        return pipe
    
    return my_pipeline

def run_example():
    # Create config
    config = Config({
        "folder_path": {
            "artifacts": "artifacts",
            "metadata": "metadata"
        }
    })
    
    # Create stack
    dev_stack = Stack(name="dev", config=config)
    
    # Create and run pipeline
    pipeline = create_example_pipeline()
    run_id = pipeline.run(dev_stack, tags={"env": "dev"})
    
    # Get run metadata
    run = dev_stack.metadata_store.get_run(run_id)
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
