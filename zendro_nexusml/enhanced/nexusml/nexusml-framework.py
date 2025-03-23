# nexusml/core/step.py
import inspect
import functools
import uuid
from typing import Any, Dict, Optional, Callable, List, Type, Union
import os
import pickle
import json

class Step:
    """Base class for all pipeline steps in NexusML."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        cache_enabled: bool = True,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[str]] = None,
    ):
        self.name = name or self.__class__.__name__
        self.cache_enabled = cache_enabled
        self.inputs = inputs or {}
        self.outputs = outputs or []
        self.step_id = str(uuid.uuid4())
        self._fn = None
        self._metadata = {}
        
    def __call__(self, *args, **kwargs):
        """Execute the step function with the given arguments."""
        if self._fn is None:
            raise ValueError("No function has been set for this step.")
        
        # Check for cached results if caching is enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(*args, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute function
        result = self._fn(*args, **kwargs)
        
        # Cache result if enabled
        if self.cache_enabled:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key based on inputs."""
        # Simple implementation - can be improved for better hashing
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{self.name}_{self.step_id}_{hash(args_str + kwargs_str)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached result if available."""
        cache_dir = os.path.join(".nexusml", "cache")
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, result: Any) -> None:
        """Save result to cache."""
        cache_dir = os.path.join(".nexusml", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for this step."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str) -> Any:
        """Get metadata for this step."""
        return self._metadata.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            "name": self.name,
            "id": self.step_id,
            "cache_enabled": self.cache_enabled,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self._metadata
        }

# Function decorator to create steps
def step(
    name: Optional[str] = None,
    cache_enabled: bool = True,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[List[str]] = None
) -> Callable:
    """Decorator to convert a function into a NexusML step."""
    
    def decorator(func: Callable) -> Step:
        step_name = name or func.__name__
        step_instance = Step(
            name=step_name,
            cache_enabled=cache_enabled,
            inputs=inputs,
            outputs=outputs
        )
        step_instance._fn = func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return step_instance(*args, **kwargs)
        
        # Attach the step instance to the wrapper for access
        wrapper.step = step_instance
        return wrapper
    
    return decorator


# nexusml/core/artifact.py
import os
import json
import pickle
from typing import Any, Dict, Optional, Union
import uuid
import datetime

class Artifact:
    """Class to represent and handle pipeline artifacts."""
    
    def __init__(
        self,
        name: str,
        data: Any,
        artifact_type: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.data = data
        self.artifact_type = artifact_type or self._infer_type(data)
        self.description = description or ""
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.now().isoformat()
    
    def _infer_type(self, data: Any) -> str:
        """Infer the type of the artifact data."""
        import numpy as np
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            return "dataframe"
        elif isinstance(data, pd.Series):
            return "series"
        elif isinstance(data, np.ndarray):
            return "numpy.ndarray"
        elif hasattr(data, 'predict'):
            return "model"
        elif isinstance(data, (dict, list)):
            return "json"
        elif isinstance(data, str):
            return "text"
        elif isinstance(data, bytes):
            return "binary"
        else:
            return "pickle"
    
    def save(self, path: Optional[str] = None) -> str:
        """Save the artifact to disk."""
        if path is None:
            path = os.path.join(".nexusml", "artifacts", self.id)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metadata
        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "name": self.name,
                "type": self.artifact_type,
                "description": self.description,
                "id": self.id,
                "created_at": self.created_at,
                "metadata": self.metadata
            }, f)
        
        # Save data based on type
        if self.artifact_type in ["json", "dict", "list"]:
            data_path = f"{path}.json"
            with open(data_path, "w") as f:
                json.dump(self.data, f)
        elif self.artifact_type == "text":
            data_path = f"{path}.txt"
            with open(data_path, "w") as f:
                f.write(self.data)
        else:
            data_path = f"{path}.pkl"
            with open(data_path, "wb") as f:
                pickle.dump(self.data, f)
        
        return path
    
    @classmethod
    def load(cls, artifact_id: str, base_path: Optional[str] = None) -> 'Artifact':
        """Load an artifact from disk."""
        if base_path is None:
            base_path = os.path.join(".nexusml", "artifacts")
        
        path = os.path.join(base_path, artifact_id)
        
        # Load metadata
        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata_info = json.load(f)
        
        # Load data based on type
        artifact_type = metadata_info["type"]
        if artifact_type in ["json", "dict", "list"]:
            data_path = f"{path}.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif artifact_type == "text":
            data_path = f"{path}.txt"
            with open(data_path, "r") as f:
                data = f.read()
        else:
            data_path = f"{path}.pkl"
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        
        # Create new artifact instance
        artifact = cls(
            name=metadata_info["name"],
            data=data,
            artifact_type=artifact_type,
            description=metadata_info["description"],
            metadata=metadata_info["metadata"]
        )
        artifact.id = metadata_info["id"]
        artifact.created_at = metadata_info["created_at"]
        
        return artifact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary representation."""
        return {
            "name": self.name,
            "type": self.artifact_type,
            "description": self.description,
            "id": self.id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


# nexusml/core/pipeline.py
from typing import Dict, List, Any, Optional, Callable, Union, Set
import uuid
import datetime
import json
import os
import networkx as nx
from .step import Step
from .artifact import Artifact

class Pipeline:
    """Class for defining and executing ML pipelines in NexusML."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or ""
        self.id = str(uuid.uuid4())
        self.steps: Dict[str, Step] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.datetime.now().isoformat()
        self._artifacts: Dict[str, Artifact] = {}
    
    def add_step(
        self,
        step: Union[Step, Callable],
        name: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Add a step to the pipeline."""
        # Handle case when step is a decorated function
        if callable(step) and hasattr(step, 'step'):
            actual_step = step.step
        elif isinstance(step, Step):
            actual_step = step
        else:
            raise TypeError("Step must be either a Step instance or a decorated function")
        
        # Set step name if provided
        if name is not None:
            actual_step.name = name
        
        # Add step to pipeline
        step_name = actual_step.name
        if step_name in self.steps:
            raise ValueError(f"Step with name '{step_name}' already exists in pipeline")
        
        self.steps[step_name] = actual_step
        self.dependencies[step_name] = dependencies or []
        
        return step_name
    
    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the pipeline with the provided inputs."""
        inputs = inputs or {}
        outputs = {}
        
        # Validate pipeline structure
        self._validate_pipeline()
        
        # Get execution order
        execution_order = self._get_execution_order()
        
        # Execute steps in order
        for step_name in execution_order:
            step = self.steps[step_name]
            
            # Prepare inputs for this step
            step_inputs = {}
            for dependency in self.dependencies[step_name]:
                if dependency in outputs:
                    # Use output from previous step
                    step_inputs[dependency] = outputs[dependency]
            
            # Add pipeline inputs if needed
            for input_name, input_value in inputs.items():
                if input_name not in step_inputs:
                    step_inputs[input_name] = input_value
            
            # Execute step
            try:
                result = step(**step_inputs)
                outputs[step_name] = result
                
                # Create artifact for result
                artifact = Artifact(
                    name=f"{step_name}_output",
                    data=result,
                    description=f"Output from step '{step_name}'"
                )
                self._artifacts[step_name] = artifact
                artifact.save()
                
            except Exception as e:
                raise RuntimeError(f"Error executing step '{step_name}': {str(e)}") from e
        
        return outputs
    
    def _validate_pipeline(self) -> None:
        """Validate pipeline structure to ensure it's a valid DAG."""
        # Check for cycles
        graph = nx.DiGraph()
        
        # Add all steps
        for step_name in self.steps:
            graph.add_node(step_name)
        
        # Add dependencies as edges
        for step_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.steps:
                    raise ValueError(f"Dependency '{dep}' for step '{step_name}' not found in pipeline")
                graph.add_edge(dep, step_name)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            raise ValueError(f"Pipeline contains cycles: {cycles}")
    
    def _get_execution_order(self) -> List[str]:
        """Determine the execution order for steps based on dependencies."""
        graph = nx.DiGraph()
        
        # Add all steps
        for step_name in self.steps:
            graph.add_node(step_name)
        
        # Add dependencies as edges
        for step_name, deps in self.dependencies.items():
            for dep in deps:
                graph.add_edge(dep, step_name)
        
        # Get topological sort
        return list(nx.topological_sort(graph))
    
    def get_artifact(self, step_name: str) -> Optional[Artifact]:
        """Get the artifact produced by a step."""
        return self._artifacts.get(step_name)
    
    def save(self, path: Optional[str] = None) -> str:
        """Save the pipeline configuration."""
        if path is None:
            path = os.path.join(".nexusml", "pipelines", f"{self.id}.json")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        pipeline_dict = {
            "name": self.name,
            "description": self.description,
            "id": self.id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "dependencies": self.dependencies,
            "artifacts": {name: artifact.to_dict() for name, artifact in self._artifacts.items()}
        }
        
        with open(path, "w") as f:
            json.dump(pipeline_dict, f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, pipeline_id: str, base_path: Optional[str] = None) -> 'Pipeline':
        """Load a pipeline from disk."""
        # This is a simplified implementation
        # A full implementation would need to recreate Step objects
        # and function references which is more complex
        
        if base_path is None:
            base_path = os.path.join(".nexusml", "pipelines")
        
        path = os.path.join(base_path, f"{pipeline_id}.json")
        
        with open(path, "r") as f:
            pipeline_dict = json.load(f)
        
        pipeline = cls(
            name=pipeline_dict["name"],
            description=pipeline_dict["description"]
        )
        pipeline.id = pipeline_dict["id"]
        pipeline.created_at = pipeline_dict["created_at"]
        pipeline.metadata = pipeline_dict["metadata"]
        pipeline.dependencies = pipeline_dict["dependencies"]
        
        # Note: This is a partial implementation
        # A complete implementation would need to recreate Step objects
        
        return pipeline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "id": self.id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "dependencies": self.dependencies
        }


# nexusml/config/base_config.py
import os
import json
from typing import Any, Dict, Optional, List, Union

class NexusConfig:
    """Base configuration class for NexusML."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        active_stack: Optional[str] = None
    ):
        self._config_dict: Dict[str, Any] = {}
        self._config_path = config_path or os.path.join(os.getcwd(), ".nexusml", "config.json")
        
        # Create default configuration
        self._create_default_config()
        
        # Load from file if it exists
        if os.path.exists(self._config_path):
            self.load()
        
        # Set active stack
        if active_stack:
            self.set_active_stack(active_stack)
    
    def _create_default_config(self) -> None:
        """Create a default configuration."""
        self._config_dict = {
            "version": "0.1.0",
            "active_stack": "default",
            "stacks": {
                "default": {
                    "storage": {
                        "type": "local",
                        "path": ".nexusml/storage"
                    },
                    "orchestrator": {
                        "type": "local"
                    },
                    "experiment_tracker": {
                        "type": "local",
                        "path": ".nexusml/experiments"
                    },
                    "model_registry": {
                        "type": "local",
                        "path": ".nexusml/models"
                    }
                }
            },
            "telemetry_enabled": False,
            "log_level": "info"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with dot notation."""
        keys = key.split(".")
        value = self._config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by key with dot notation."""
        keys = key.split(".")
        config = self._config_dict
        
        # Traverse to the right level
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_active_stack(self) -> str:
        """Get the name of the active stack."""
        return self._config_dict.get("active_stack", "default")
    
    def set_active_stack(self, stack_name: str) -> None:
        """Set the active stack."""
        if stack_name not in self._config_dict.get("stacks", {}):
            raise ValueError(f"Stack '{stack_name}' does not exist")
        
        self._config_dict["active_stack"] = stack_name
    
    def get_stack_config(self, stack_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific stack."""
        stack = stack_name or self.get_active_stack()
        return self._config_dict.get("stacks", {}).get(stack, {})
    
    def create_stack(self, stack_name: str, config: Dict[str, Any]) -> None:
        """Create a new stack configuration."""
        if "stacks" not in self._config_dict:
            self._config_dict["stacks"] = {}
        
        if stack_name in self._config_dict["stacks"]:
            raise ValueError(f"Stack '{stack_name}' already exists")
        
        self._config_dict["stacks"][stack_name] = config
    
    def update_stack(self, stack_name: str, config: Dict[str, Any]) -> None:
        """Update an existing stack configuration."""
        if "stacks" not in self._config_dict or stack_name not in self._config_dict["stacks"]:
            raise ValueError(f"Stack '{stack_name}' does not exist")
        
        self._config_dict["stacks"][stack_name].update(config)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to a file."""
        save_path = path or self._config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(self._config_dict, f, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load configuration from a file."""
        load_path = path or self._config_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Configuration file not found: {load_path}")
        
        with open(load_path, "r") as f:
            loaded_config = json.load(f)
        
        self._config_dict.update(loaded_config)
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._create_default_config()


# nexusml/cli/cli.py
import click
import os
import json
from typing import Dict, Any
import sys
import importlib.util
from ..config.base_config import NexusConfig

@click.group()
def cli():
    """NexusML - Machine Learning Pipeline Framework."""
    pass

@cli.command()
@click.option('--path', default=None, help='Directory to initialize NexusML in')
def init(path):
    """Initialize a new NexusML project."""
    project_dir = path or os.getcwd()
    
    # Create project structure
    structures = [
        ".nexusml",
        ".nexusml/storage",
        ".nexusml/cache",
        ".nexusml/artifacts",
        ".nexusml/pipelines",
        ".nexusml/models",
        ".nexusml/experiments"
    ]
    
    for structure in structures:
        os.makedirs(os.path.join(project_dir, structure), exist_ok=True)
    
    # Create default configuration
    config = NexusConfig()
    config.save(os.path.join(project_dir, ".nexusml", "config.json"))
    
    click.echo(f"Initialized NexusML project in {project_dir}")

@cli.command()
@click.argument('pipeline_file')
@click.option('--params', default=None, help='JSON file with pipeline parameters')
def run(pipeline_file, params):
    """Run a NexusML pipeline."""
    # Load pipeline file (Python file containing pipeline definition)
    if not os.path.exists(pipeline_file):
        click.echo(f"Pipeline file not found: {pipeline_file}", err=True)
        return
    
    # Load parameters if provided
    pipeline_params = {}
    if params:
        if not os.path.exists(params):
            click.echo(f"Parameters file not found: {params}", err=True)
            return
        
        with open(params, 'r') as f:
            pipeline_params = json.load(f)
    
    # Execute pipeline file
    try:
        # Add directory to sys.path
        sys.path.append(os.path.dirname(os.path.abspath(pipeline_file)))
        
        # Load module
        spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_file)
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        # Check if module has a run function
        if hasattr(pipeline_module, 'run'):
            result = pipeline_module.run(pipeline_params)
            click.echo("Pipeline executed successfully")
            click.echo(f"Result: {result}")
        else:
            click.echo("Pipeline file must contain a 'run' function", err=True)
    except Exception as e:
        click.echo(f"Error executing pipeline: {str(e)}", err=True)
        import traceback
        click.echo(traceback.format_exc(), err=True)

@cli.command()
@click.option('--name', default=None, help='Name of the stack to create')
def create_stack(name):
    """Create a new configuration stack."""
    if not name:
        click.echo("Stack name is required", err=True)
        return
    
    config = NexusConfig()
    
    # Check if stack already exists
    if name in config.get("stacks", {}):
        click.echo(f"Stack '{name}' already exists", err=True)
        return
    
    # Create new stack with default configuration
    config.create_stack(name, {
        "storage": {
            "type": "local",
            "path": f".nexusml/storage/{name}"
        },
        "orchestrator": {
            "type": "local"
        },
        "experiment_tracker": {
            "type": "local",
            "path": f".nexusml/experiments/{name}"
        },
        "model_registry": {
            "type": "local",
            "path": f".nexusml/models/{name}"
        }
    })
    
    config.save()
    click.echo(f"Created new stack: {name}")

@cli.command()
@click.option('--name', default=None, help='Name of the stack to activate')
def use_stack(name):
    """Set the active stack."""
    if not name:
        click.echo("Stack name is required", err=True)
        return
    
    config = NexusConfig()
    
    # Check if stack exists
    if name not in config.get("stacks", {}):
        click.echo(f"Stack '{name}' does not exist", err=True)
        return
    
    # Set active stack
    config.set_active_stack(name)
    config.save()
    click.echo(f"Activated stack: {name}")

@cli.command()
def list_stacks():
    """List all available stacks."""
    config = NexusConfig()
    stacks = config.get("stacks", {})
    active_stack = config.get_active_stack()
    
    click.echo("Available stacks:")
    for stack_name in stacks:
        if stack_name == active_stack:
            click.echo(f"* {stack_name} (active)")
        else:
            click.echo(f"  {stack_name}")

def main():
    """Entry point for the CLI."""
    cli()

if __name__ == "__main__":
    main()



# example_pipeline.py
from nexusml.core.step import step
from nexusml.core.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define steps using decorators
@step(name="load_data", outputs=["data"])
def load_data(data_path):
    """Load data from a CSV file."""
    data = pd.read_csv(data_path)
    return data

@step(name="preprocess", outputs=["X", "y"])
def preprocess(data):
    """Preprocess the data and split features and target."""
    # Example preprocessing
    data = data.dropna()
    
    # Extract features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    return X, y

@step(name="split_data", outputs=["X_train", "X_test", "y_train", "y_test"])
def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

@step(name="train_model", outputs=["model"])
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

@step(name="evaluate", outputs=["accuracy", "predictions"])
def evaluate(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

# Define the pipeline
def create_pipeline():
    """Create and return the ML pipeline."""
    pipeline = Pipeline(name="iris_classification", description="Classify Iris flowers")
    
    # Add steps to the pipeline
    pipeline.add_step(load_data)
    pipeline.add_step(preprocess, dependencies=["load_data"])
    pipeline.add_step(split_data, dependencies=["preprocess"])
    pipeline.add_step(train_model, dependencies=["split_data"])
    pipeline.add_step(evaluate, dependencies=["train_model", "split_data"])
    
    return pipeline

def run(params=None):
    """Run the pipeline with the given parameters."""
    params = params or {}
    
    # Default parameters
    default_params = {
        "data_path": "iris.csv",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100
    }
    
    # Update with provided parameters
    default_params.update(params)
    
    # Create and run the pipeline
    pipeline = create_pipeline()
    results = pipeline.run(default_params)
    
    # Save pipeline configuration
    pipeline.save()
    
    # Extract an