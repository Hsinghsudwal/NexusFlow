# Project Structure
"""
zen_framework/
├── README.md
├── pyproject.toml
├── setup.py
├── zen_framework/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration management
│   ├── steps/
│   │   ├── __init__.py
│   │   └── base_step.py        # Base step class (like ZenML)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── base_pipeline.py    # Base pipeline class
│   ├── materializers/
│   │   ├── __init__.py
│   │   ├── base.py             # Base materializer
│   │   └── pandas.py           # Pandas materializer
│   ├── stacks/
│   │   ├── __init__.py
│   │   ├── stack.py            # Stack definition (ZenML style)
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Base component
│   │   │   ├── experiment_tracker.py
│   │   │   ├── orchestrator.py
│   │   │   ├── artifact_store.py
│   │   │   └── model_deployer.py
│   │   └── stack_registry.py   # Track available stacks
│   ├── artifacts/
│   │   ├── __init__.py
│   │   └── artifact_store.py   # Artifact storage
│   ├── metadata/
│   │   ├── __init__.py
│   │   └── metadata_store.py   # Metadata storage
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualizer.py       # Pipeline visualization
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # Logging utilities
│       └── decorators.py       # Useful decorators
└── example/
    ├── configs/                # Stack and pipeline configs
    │   ├── local_stack.yaml
    │   └── production_stack.yaml
    ├── data/                   # Data storage
    │   ├── raw/
    │   └── processed/
    ├── steps/                  # Step definitions
    │   ├── __init__.py
    │   ├── data_loader.py
    │   ├── preprocessor.py
    │   └── model_trainer.py
    ├── pipelines/              # Pipeline definitions
    │   ├── __init__.py
    │   └── training_pipeline.py
    └── run.py                  # Main entry point
"""

# Core Components - ZenML Style Approach

import os
import inspect
import yaml
import uuid
import datetime
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Type
from abc import ABC, abstractmethod


# 1. Step - ZenML's core execution unit
class BaseStep(ABC):
    """
    A step represents a single data processing unit in a pipeline.
    Similar to ZenML's step concept.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.uuid = str(uuid.uuid4())
        self._output_artifacts = {}
        self._input_artifacts = {}
        
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the step logic."""
        pass
        
    def __call__(self, *args, **kwargs) -> Any:
        """Make the step callable."""
        # This would normally capture inputs and outputs for the metadata store
        # but simplified here
        start_time = datetime.datetime.now()
        result = self.process(*args, **kwargs)
        end_time = datetime.datetime.now()
        
        # Log execution time
        execution_time = (end_time - start_time).total_seconds()
        print(f"Step '{self.name}' executed in {execution_time:.2f} seconds")
        
        return result
        
    def connect(self, **connections):
        """Connect this step to other steps."""
        self._input_artifacts.update(connections)
        return self
        

# 2. Functional step decorator - ZenML style
def step(func=None, *, name=None, materializers=None):
    """Decorator to convert a function into a Step."""
    
    def decorator(func):
        # Extract information from function signature
        sig = inspect.signature(func)
        
        # Create a new step class dynamically
        class FunctionStep(BaseStep):
            def __init__(self):
                super().__init__(name=name or func.__name__)
                self.func = func
                self.materializers = materializers or {}
                
            def process(self, *args, **kwargs):
                # Execute the function
                return self.func(*args, **kwargs)
                
        # Return an instance of the new step class
        return FunctionStep()
        
    # Handle both @step and @step() forms
    if func is None:
        return decorator
    return decorator(func)


# 3. Pipeline - Similar to ZenML's pipeline concept
class Pipeline:
    """
    A pipeline is a directed acyclic graph of steps.
    """
    
    def __init__(self, name: str, steps: Optional[List[BaseStep]] = None):
        self.name = name
        self.steps = steps or []
        self.uuid = str(uuid.uuid4())
        self._step_dependencies = {}
        
    def add_step(self, step: BaseStep) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
        
    def run(self, context):
        """Run the pipeline with a given context."""
        results = {}
        
        # In ZenML-like systems, the orchestrator would handle this
        # This is a simple sequential execution
        for step in self.steps:
            # Prepare inputs from previous steps if needed
            kwargs = {}
            for param, artifact_key in step._input_artifacts.items():
                if artifact_key in results:
                    kwargs[param] = results[artifact_key]
                    
            # Execute the step
            result = step(**kwargs)
            
            # Store the result
            results[step.name] = result
            
        return results


# 4. Stack Component base class - ZenML style
class StackComponent(ABC):
    """Base class for all stack components."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        pass
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# 5. Stack - ZenML's approach to MLOps integrations
class Stack:
    """
    Container for MLOps stack components.
    Similar to ZenML's stack concept.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.uuid = str(uuid.uuid4())
        self.components = {}
        
    def add_component(self, component_type: str, component: StackComponent):
        """Add a component to the stack."""
        self.components[component_type] = component
        return self
        
    def get_component(self, component_type: str) -> Optional[StackComponent]:
        """Get a component from the stack by type."""
        return self.components.get(component_type)
        
    def initialize(self):
        """Initialize all components in the stack."""
        for component in self.components.values():
            component.initialize()
            
    @classmethod
    def load(cls, stack_path: str) -> "Stack":
        """Load a stack configuration from a YAML file."""
        with open(stack_path, 'r') as f:
            config = yaml.safe_load(f)
            
        stack = cls(name=config.get('name', 'default_stack'))
        
        # Add components based on configuration
        # This would normally involve a registry of component types
        # but simplified here
        
        return stack
        
    def save(self, stack_path: str):
        """Save stack configuration to a YAML file."""
        config = {
            'name': self.name,
            'uuid': self.uuid,
            'components': {
                component_type: {
                    'name': component.name,
                    'type': component.__class__.__name__,
                    'config': component.config
                }
                for component_type, component in self.components.items()
            }
        }
        
        with open(stack_path, 'w') as f:
            yaml.dump(config, f)


# 6. Stack Registry (ZenML has a similar concept)
class StackRegistry:
    """Registry to keep track of available stacks."""
    
    def __init__(self):
        self.stacks = {}
        self.active_stack = None
        
    def register_stack(self, stack: Stack):
        """Register a stack."""
        self.stacks[stack.name] = stack
        
    def get_stack(self, name: str) -> Optional[Stack]:
        """Get a stack by name."""
        return self.stacks.get(name)
        
    def set_active_stack(self, name: str):
        """Set the active stack."""
        if name not in self.stacks:
            raise ValueError(f"Stack '{name}' not found in registry")
            
        self.active_stack = self.stacks[name]
        
    def get_active_stack(self) -> Optional[Stack]:
        """Get the active stack."""
        return self.active_stack


# 7. MLOps Stack Component Implementations (ZenML style)

# Experiment Tracker Component
class ExperimentTracker(StackComponent):
    """MLflow-based experiment tracking component."""
    
    def initialize(self):
        """Initialize MLflow tracking."""
        try:
            import mlflow
            self.mlflow = mlflow
            
            # Configure MLflow
            tracking_uri = self.config.get('tracking_uri')
            experiment_name = self.config.get('experiment_name', 'default')
            
            if tracking_uri:
                self.mlflow.set_tracking_uri(tracking_uri)
                
            self.mlflow.set_experiment(experiment_name)
            
        except ImportError:
            print("Warning: MLflow not installed. Experiment tracking disabled.")
            self.mlflow = None
            
    def start_run(self, run_name=None):
        """Start a new run."""
        if self.mlflow:
            return self.mlflow.start_run(run_name=run_name)
        return DummyContext()
        
    def log_params(self, params):
        """Log parameters."""
        if self.mlflow:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
                
    def log_metrics(self, metrics):
        """Log metrics."""
        if self.mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value)
                
    def log_model(self, model, name):
        """Log a model."""
        if self.mlflow:
            self.mlflow.sklearn.log_model(model, name)


# Orchestrator Component
class Orchestrator(StackComponent):
    """Pipeline orchestration component."""
    
    def initialize(self):
        """Initialize the orchestrator."""
        orchestrator_type = self.config.get('type', 'local')
        
        if orchestrator_type == 'airflow':
            try:
                from airflow import DAG
                from airflow.utils.dates import days_ago
                self.dag_factory = lambda name: DAG(
                    name,
                    default_args={'owner': 'zen_framework'},
                    schedule_interval=None,
                    start_date=days_ago(1)
                )
            except ImportError:
                print("Warning: Airflow not installed. Using local orchestration.")
                orchestrator_type = 'local'
                
        elif orchestrator_type == 'prefect':
            try:
                import prefect
                from prefect import Flow
                self.flow_factory = lambda name: Flow(name)
            except ImportError:
                print("Warning: Prefect not installed. Using local orchestration.")
                orchestrator_type = 'local'
                
        self.orchestrator_type = orchestrator_type
        
    def run_pipeline(self, pipeline, context):
        """Run a pipeline with the configured orchestrator."""
        if self.orchestrator_type == 'local':
            # Simple sequential execution
            return pipeline.run(context)
            
        elif self.orchestrator_type == 'airflow':
            # Create DAG
            dag = self.dag_factory(pipeline.name)
            # This would generate Airflow tasks
            print(f"Would run pipeline '{pipeline.name}' with Airflow")
            return None
            
        elif self.orchestrator_type == 'prefect':
            # Create Flow
            flow = self.flow_factory(pipeline.name)
            # This would generate Prefect tasks
            print(f"Would run pipeline '{pipeline.name}' with Prefect")
            return None


# Artifact Store Component
class ArtifactStore(StackComponent):
    """Storage component for artifacts."""
    
    def initialize(self):
        """Initialize the artifact store."""
        store_type = self.config.get('type', 'local')
        base_path = self.config.get('path', './artifacts')
        
        if store_type == 's3':
            try:
                import boto3
                self.client = boto3.client('s3')
                self.bucket = self.config.get('bucket')
            except ImportError:
                print("Warning: boto3 not installed. Using local artifact store.")
                store_type = 'local'
                
        elif store_type == 'gcs':
            try:
                from google.cloud import storage
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.config.get('bucket'))
            except ImportError:
                print("Warning: google-cloud-storage not installed. Using local artifact store.")
                store_type = 'local'
                
        self.store_type = store_type
        self.base_path = base_path
        
        # Create local directory if needed
        if store_type == 'local' and not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            
    def save_artifact(self, artifact, name):
        """Save an artifact."""
        if self.store_type == 'local':
            import pickle
            path = os.path.join(self.base_path, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(artifact, f)
                
            return path
            
        elif self.store_type == 's3':
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as temp:
                pickle.dump(artifact, temp)
                temp.flush()
                path = os.path.join(self.base_path, name)
                self.client.upload_file(temp.name, self.bucket, path)
                
            return f"s3://{self.bucket}/{path}"
            
        elif self.store_type == 'gcs':
            import pickle
            
            path = os.path.join(self.base_path, name)
            blob = self.bucket.blob(path)
            blob.upload_from_string(pickle.dumps(artifact))
            
            return f"gs://{self.bucket.name}/{path}"
            
    def load_artifact(self, name):
        """Load an artifact."""
        if self.store_type == 'local':
            import pickle
            path = os.path.join(self.base_path, name)
            
            with open(path, 'rb') as f:
                return pickle.load(f)
                
        elif self.store_type == 's3':
            import pickle
            import tempfile
            
            path = os.path.join(self.base_path, name)
            with tempfile.NamedTemporaryFile() as temp:
                self.client.download_file(self.bucket, path, temp.name)
                with open(temp.name, 'rb') as f:
                    return pickle.load(f)
                    
        elif self.store_type == 'gcs':
            import pickle
            
            path = os.path.join(self.base_path, name)
            blob = self.bucket.blob(path)
            
            return pickle.loads(blob.download_as_bytes())


# Model Deployer Component
class ModelDeployer(StackComponent):
    """Model deployment component."""
    
    def initialize(self):
        """Initialize the model deployer."""
        self.deployer_type = self.config.get('type', 'bentoml')
        
    def deploy_model(self, model, name, version=None):
        """Deploy a model."""
        version = version or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        if self.deployer_type == 'bentoml':
            try:
                import bentoml
                bentoml.save(model, name)
                print(f"Model {name} saved to BentoML repository")
                return f"bentoml://{name}/{version}"
            except ImportError:
                print("Warning: BentoML not installed. Deployment disabled.")
                return None
                
        elif self.deployer_type == 'mlflow':
            try:
                import mlflow
                mlflow.sklearn.log_model(model, name)
                print(f"Model {name} logged to MLflow")
                return f"mlflow://{mlflow.active_run().info.run_id}/{name}"
            except ImportError:
                print("Warning: MLflow not installed. Deployment disabled.")
                return None


# 8. Materializers - Similar to ZenML's concept
class Materializer(ABC):
    """Base class for materializers that handle artifact serialization."""
    
    @abstractmethod
    def save(self, artifact, path: str) -> None:
        """Save an artifact to a path."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> Any:
        """Load an artifact from a path."""
        pass


class PandasMaterializer(Materializer):
    """Materializer for pandas DataFrames."""
    
    def save(self, df, path: str) -> None:
        """Save a DataFrame to a path."""
        import pandas as pd
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Determine format from file extension
        if path.endswith('.csv'):
            df.to_csv(path, index=False)
        elif path.endswith('.parquet'):
            df.to_parquet(path, index=False)
        elif path.endswith('.pkl'):
            df.to_pickle(path)
        else:
            # Default to parquet
            df.to_parquet(f"{path}.parquet", index=False)
            
    def load(self, path: str) -> Any:
        """Load a DataFrame from a path."""
        import pandas as pd
        
        # Determine format from file extension
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.pkl'):
            return pd.read_pickle(path)
        else:
            # Try to infer format or default to parquet
            try:
                return pd.read_parquet(f"{path}.parquet")
            except:
                raise ValueError(f"Cannot determine format for {path}")


# 9. Pipeline Context - Execution environment
class Context:
    """Execution context for pipelines."""
    
    def __init__(self, stack: Stack = None, params: Dict[str, Any] = None):
        self.stack = stack
        self.params = params or {}
        self.artifacts = {}
        
    def get_param(self, name: str, default=None) -> Any:
        """Get a parameter from the context."""
        return self.params.get(name, default)
        
    def set_artifact(self, name: str, artifact: Any) -> None:
        """Set an artifact in the context."""
        self.artifacts[name] = artifact
        
        # Save to artifact store if available
        if self.stack and 'artifact_store' in self.stack.components:
            store = self.stack.get_component('artifact_store')
            store.save_artifact(artifact, name)
            
    def get_artifact(self, name: str) -> Any:
        """Get an artifact from the context."""
        if name in self.artifacts:
            return self.artifacts[name]
            
        # Try to load from artifact store
        if self.stack and 'artifact_store' in self.stack.components:
            store = self.stack.get_component('artifact_store')
            try:
                artifact = store.load_artifact(name)
                self.artifacts[name] = artifact
                return artifact
            except:
                pass
                
        return None


# 10. Pipeline Runner - Simple execution engine
class PipelineRunner:
    """Runner for executing pipelines."""
    
    def __init__(self, stack: Stack = None):
        self.stack = stack
        
    def run(self, pipeline: Pipeline, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a pipeline."""
        # Create context
        context = Context(stack=self.stack, params=params or {})
        
        # Use orchestrator if available
        if self.stack and 'orchestrator' in self.stack.components:
            orchestrator = self.stack.get_component('orchestrator')
            return orchestrator.run_pipeline(pipeline, context)
        else:
            # Simple local execution
            return pipeline.run(context)


# 11. Dummy context manager for MLflow fallback
class DummyContext:
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 12. CLI commands
def create_cli():
    """Create a CLI for the framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZenML-like Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="Pipeline to run")
    run_parser.add_argument("--stack", help="Stack to use")
    
    # Stack commands
    stack_parser = subparsers.add_parser("stack", help="Stack operations")
    stack_subparsers = stack_parser.add_subparsers(dest="stack_command")
    
    # Register stack
    register_parser = stack_subparsers.add_parser("register", help="Register a stack")
    register_parser.add_argument("config", help="Stack configuration file")
    
    # List stacks
    list_parser = stack_subparsers.add_parser("list", help="List registered stacks")
    
    # Set active stack
    activate_parser = stack_subparsers.add_parser("activate", help="Set the active stack")
    activate_parser.add_argument("name", help="Stack name")
    
    return parser


# 13. Example Usage - How to use the framework ZenML style
def example_mlops_pipeline():
    """Example of how to use the framework in ZenML style."""
    # 1. Define steps using the decorator
    @step
    def load_data(data_path: str) -> Any:
        """Load data from a file."""
        import pandas as pd
        return pd.read_csv(data_path)
    
    @step
    def preprocess(data):
        """Preprocess the data."""
        # Drop missing values
        data = data.dropna()
        
        # Convert categorical variables
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category').cat.codes
            
        return data
    
    @step
    def split_data(data, test_size: float = 0.2):
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        # Assume the last column is the target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    @step
    def train_model(X_train, y_train, n_estimators: int = 100):
        """Train a random forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    @step
    def evaluate_model(model, X_test, y_test):
        """Evaluate the model."""
        from sklearn.metrics import accuracy_score
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {"accuracy": accuracy}
    
    # 2. Define a pipeline
    training_pipeline = Pipeline(name="train_classifier")
    
    # 3. Add steps to the pipeline with connections
    # This is more like ZenML's style of connecting steps
    data = load_data().connect(data_path="data/raw/data.csv")
    processed_data = preprocess().connect(data=data)
    X_train, X_test, y_train, y_test = split_data().connect(data=processed_data, test_size=0.2)
    model = train_model().connect(X_train=X_train, y_train=y_train, n_estimators=100)
    metrics = evaluate_model().connect(model=model, X_test=X_test, y_test=y_test)
    
    # Add all steps to the pipeline
    training_pipeline.add_step(data)
    training_pipeline.add_step(processed_data)
    training_pipeline.add_step(split_data)
    training_pipeline.add_step(model)
    training_pipeline.add_step(metrics)
    
    return training_pipeline


# 14. Creating and running with a stack
def run_example_pipeline():
    """Run the example pipeline with a stack."""
    # 1. Create stack components
    experiment_tracker = ExperimentTracker(
        name="mlflow_tracker",
        config={
            "tracking_uri": "sqlite:///mlruns.db",
            "experiment_name": "model_training"
        }
    )
    
    artifact_store = ArtifactStore(
        name="local_store",
        config={
            "type": "local",
            "path": "./artifacts"
        }
    )
    
    orchestrator = Orchestrator(
        name="local_orchestrator",
        config={
            "type": "local"
        }
    )
    
    model_deployer = ModelDeployer(
        name="bentoml_deployer",
        config={
            "type": "bentoml"
        }
    )
    
    # 2. Create and initialize stack
    stack = Stack(name="local_dev_stack")
    stack.add_component("experiment_tracker", experiment_tracker)
    stack.add_component("artifact_store", artifact_store)
    stack.add_component("orchestrator", orchestrator)
    stack.add_component("model_deployer", model_deployer)
    stack.initialize()
    
    # 3. Register the stack
    registry = StackRegistry()
    registry.register_stack(stack)
    registry.set_active_stack("local_dev_stack")
    
    # 4. Create pipeline
    pipeline = example_mlops_pipeline()
    
    # 5. Create runner with stack
    runner = PipelineRunner(stack=stack)
    
    # 6. Run pipeline
    results = runner.run(
        pipeline=pipeline,
        params={
            "n_estimators": 200,
            "test_size": 0.25
        }
    )
    
    return results


# Example of saving and loading stacks
def stack_config_example():
    """Example of saving and loading stack configurations."""
    # Create a stack
    stack = Stack(name="production_stack")
    
    # Add components
    stack.add_component(
        "experiment_tracker",
        ExperimentTracker(
            name="mlflow_remote",
            config={
                "tracking_uri": "https://mlflow.example.com",
                "experiment_name": "production_models"
            }
        )
    )
    
    stack.add_component(
        "artifact_store",
        ArtifactStore(
            name="s3_store",
            config={
                "type": "s3",
                "bucket": "ml-artifacts",
                "path": "production"
            }
        )
    )
    
    # Save stack configuration
    stack.save("./configs/production_stack.yaml")
    
    # Load stack configuration
    loaded_stack = Stack.load("./configs/production_stack.yaml")
    
    return loaded_stack


# Main function
if __name__ == "__main__":
    print("ZenML-like Framework Example")
    results = run_example_pipeline()
    print("Pipeline execution results:", results)
