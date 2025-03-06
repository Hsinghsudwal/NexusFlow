# Project Structure
"""
ml_framework/
├── README.md
├── pyproject.toml
├── setup.py
├── ml_framework/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py        # Framework configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── context.py         # Pipeline execution context
│   │   ├── pipeline.py        # Pipeline definition
│   │   ├── node.py            # Individual pipeline nodes/steps
│   │   └── registry.py        # Component registry
│   ├── io/
│   │   ├── __init__.py
│   │   ├── catalog.py         # Data catalog for managing datasets
│   │   └── data_manager.py    # Data access layer
│   ├── stack/
│   │   ├── __init__.py
│   │   ├── stack.py           # MLOps stack definition
│   │   ├── base.py            # Base stack component
│   │   ├── tracking.py        # Experiment tracking (MLflow)
│   │   ├── orchestration.py   # Workflow orchestration (Airflow/Prefect)
│   │   ├── storage.py         # Storage components (S3, GCS)
│   │   ├── versioning.py      # Model & data versioning (DVC)
│   │   └── deployment.py      # Model deployment (BentoML/TF Serving)
│   ├── visualize/
│   │   ├── __init__.py
│   │   └── pipeline_viz.py    # Pipeline visualization
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # Logging utilities
│       └── decorators.py      # Useful decorators
└── example/
    ├── conf/                  # Project configuration
    │   ├── base/              # Base configurations
    │   ├── local/             # Local environment configs
    │   └── prod/              # Production environment configs
    ├── data/                  # Data storage
    │   ├── 01_raw/            # Raw data
    │   ├── 02_intermediate/   # Intermediate processed data
    │   ├── 03_primary/        # Primary processed data
    │   ├── 04_feature/        # Feature data
    │   ├── 05_model_input/    # Model input data
    │   ├── 06_models/         # Model storage
    │   └── 07_model_output/   # Model output data
    ├── notebooks/             # Exploratory notebooks
    ├── pipelines/             # Pipeline definitions
    │   ├── __init__.py
    │   ├── data_processing/
    │   ├── feature_engineering/
    │   └── training/
    └── main.py                # Main entry point
"""

# Core Framework Components

# 1. Node (similar to Kedro's node)
import inspect
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional


class Node:
    """
    A node represents a single step in a data pipeline.
    """
    
    def __init__(
        self,
        func: Callable,
        inputs: Union[None, str, List[str], Dict[str, str]] = None,
        outputs: Union[None, str, List[str], Dict[str, str]] = None,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ):
        self.func = func
        self.name = name or func.__name__
        self.tags = tags or set()
        
        # Handle inputs
        if inputs is None:
            self.inputs = {}
        elif isinstance(inputs, str):
            self.inputs = {inputs: inputs}
        elif isinstance(inputs, list):
            self.inputs = {i: i for i in inputs}
        else:
            self.inputs = inputs
            
        # Handle outputs
        if outputs is None:
            self.outputs = {}
        elif isinstance(outputs, str):
            self.outputs = {outputs: outputs}
        elif isinstance(outputs, list):
            self.outputs = {o: o for o in outputs}
        else:
            self.outputs = outputs
            
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the node's function."""
        inputs = inputs or {}
        
        # Map inputs to the function's parameters
        params = {}
        for param_name, catalog_name in self.inputs.items():
            params[param_name] = inputs.get(catalog_name)
            
        # Execute the function
        result = self.func(**params)
        
        # Handle different return types
        if len(self.outputs) == 0:
            return {}
        elif len(self.outputs) == 1 and not isinstance(result, tuple):
            output_name = list(self.outputs.keys())[0]
            return {self.outputs[output_name]: result}
        else:
            if not isinstance(result, tuple):
                result = (result,)
            return {self.outputs[k]: v for k, v in zip(self.outputs.keys(), result)}
            
    def __repr__(self):
        return f"Node(name='{self.name}', inputs={self.inputs}, outputs={self.outputs})"


# 2. Pipeline (similar to both Kedro and ZenML pipeline)
class Pipeline:
    """
    A pipeline is a collection of nodes that are executed in order.
    """
    
    def __init__(
        self,
        nodes: List[Node],
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ):
        self.nodes = nodes
        self.name = name or "pipeline"
        self.tags = tags or set()
        self._validate_pipeline()
        
    def _validate_pipeline(self):
        """Validate that the pipeline is well-formed."""
        # Check for circular dependencies
        all_inputs = set()
        all_outputs = set()
        
        for node in self.nodes:
            all_inputs.update(node.inputs.values())
            all_outputs.update(node.outputs.values())
            
        # Check for missing inputs (except for external inputs)
        missing_inputs = all_inputs - all_outputs
        # This would typically have more complex logic for external inputs
        
    def only_nodes_with_tags(self, tags: Set[str]) -> "Pipeline":
        """Return a new pipeline with only nodes that have the specified tags."""
        nodes = [node for node in self.nodes if tags.intersection(node.tags)]
        return Pipeline(nodes=nodes, name=self.name)
    
    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Combine two pipelines."""
        if not isinstance(other, Pipeline):
            return NotImplemented
            
        return Pipeline(
            nodes=self.nodes + other.nodes,
            name=f"{self.name}+{other.name}"
        )


# 3. Data Catalog (similar to Kedro's DataCatalog)
class DataCatalog:
    """
    A data catalog manages the loading and saving of datasets.
    """
    
    def __init__(self):
        self._datasets = {}
        
    def add(self, name: str, dataset):
        """Add a dataset to the catalog."""
        self._datasets[name] = dataset
        
    def load(self, name: str) -> Any:
        """Load a dataset from the catalog."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found in the catalog")
            
        return self._datasets[name].load()
        
    def save(self, name: str, data: Any):
        """Save data to a dataset in the catalog."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found in the catalog")
            
        self._datasets[name].save(data)
        
    def __contains__(self, name: str) -> bool:
        return name in self._datasets


# 4. Dataset (base class for different types of data sources)
class Dataset:
    """Base class for all datasets."""
    
    def load(self) -> Any:
        """Load data from the dataset."""
        raise NotImplementedError
        
    def save(self, data: Any):
        """Save data to the dataset."""
        raise NotImplementedError


# 5. Context (execution context for pipelines)
class Context:
    """
    Execution context for pipelines.
    """
    
    def __init__(
        self,
        catalog: DataCatalog,
        stack = None,  # MLOps stack
        params: Dict[str, Any] = None
    ):
        self.catalog = catalog
        self.stack = stack
        self.params = params or {}
        
    def run(self, pipeline: Pipeline, inputs: Dict[str, Any] = None):
        """Execute a pipeline with the current context."""
        inputs = inputs or {}
        required_outputs = set()
        outputs = {}
        
        # Discover required outputs for the whole pipeline
        for node in pipeline.nodes:
            required_outputs.update(node.inputs.values())
            
        # Add inputs to outputs
        outputs.update(inputs)
        
        # Execute each node in order
        for node in pipeline.nodes:
            node_inputs = {}
            
            # Gather inputs for this node
            for param_name, catalog_name in node.inputs.items():
                if catalog_name in outputs:
                    node_inputs[param_name] = outputs[catalog_name]
                elif catalog_name in self.catalog:
                    node_inputs[param_name] = self.catalog.load(catalog_name)
                else:
                    raise ValueError(f"Input '{catalog_name}' for node '{node.name}' not found")
                    
            # Execute the node
            if self.stack and hasattr(self.stack, 'tracking'):
                with self.stack.tracking.start_run(node.name):
                    node_outputs = node.run(node_inputs)
            else:
                node_outputs = node.run(node_inputs)
                
            # Store outputs
            outputs.update(node_outputs)
            
            # Save outputs to catalog if needed
            for catalog_name in node.outputs.values():
                if catalog_name in self.catalog:
                    self.catalog.save(catalog_name, outputs[catalog_name])
                    
        return outputs


# 6. MLOps Stack Components
class StackComponent:
    """Base class for all stack components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.initialize()
        
    def initialize(self):
        """Initialize the component."""
        pass
        
        
class Stack:
    """
    Container for MLOps stack components.
    """
    
    def __init__(self):
        self.components = {}
        
    def add_component(self, name: str, component: StackComponent):
        """Add a component to the stack."""
        self.components[name] = component
        setattr(self, name, component)
        
    def __getattr__(self, name):
        if name in self.components:
            return self.components[name]
        raise AttributeError(f"Stack has no component '{name}'")


# 7. Experiment Tracking Component
class ExperimentTracker(StackComponent):
    """
    MLflow-based experiment tracking component.
    """
    
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
            
    def start_run(self, name=None):
        """Start a new run."""
        if self.mlflow:
            return self.mlflow.start_run(run_name=name)
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


# 8. Model Registry Component
class ModelRegistry(StackComponent):
    """
    Model versioning and registry component.
    """
    
    def initialize(self):
        """Initialize model registry."""
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            print("Warning: MLflow not installed. Model registry disabled.")
            self.mlflow = None
            
    def register_model(self, model_uri, name):
        """Register a model in the registry."""
        if self.mlflow:
            return self.mlflow.register_model(model_uri, name)
        return None
        
    def get_model(self, name, stage="Production"):
        """Get a model from the registry."""
        if self.mlflow:
            return self.mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
        return None


# 9. Storage Component
class StorageComponent(StackComponent):
    """
    Storage component for different backends (S3, GCS, local).
    """
    
    def initialize(self):
        """Initialize storage."""
        storage_type = self.config.get('type', 'local')
        
        if storage_type == 's3':
            try:
                import boto3
                self.client = boto3.client('s3')
                self.bucket = self.config.get('bucket')
            except ImportError:
                print("Warning: boto3 not installed. S3 storage disabled.")
                storage_type = 'local'
                
        elif storage_type == 'gcs':
            try:
                from google.cloud import storage
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.config.get('bucket'))
            except ImportError:
                print("Warning: google-cloud-storage not installed. GCS storage disabled.")
                storage_type = 'local'
                
        self.storage_type = storage_type
        
    def save(self, data, path):
        """Save data to storage."""
        if self.storage_type == 'local':
            import os
            import pickle
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
                
        elif self.storage_type == 's3':
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as temp:
                pickle.dump(data, temp)
                temp.flush()
                self.client.upload_file(temp.name, self.bucket, path)
                
        elif self.storage_type == 'gcs':
            import pickle
            
            blob = self.bucket.blob(path)
            blob.upload_from_string(pickle.dumps(data))
            
    def load(self, path):
        """Load data from storage."""
        if self.storage_type == 'local':
            import pickle
            
            with open(path, 'rb') as f:
                return pickle.load(f)
                
        elif self.storage_type == 's3':
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as temp:
                self.client.download_file(self.bucket, path, temp.name)
                with open(temp.name, 'rb') as f:
                    return pickle.load(f)
                    
        elif self.storage_type == 'gcs':
            import pickle
            
            blob = self.bucket.blob(path)
            return pickle.loads(blob.download_as_bytes())


# 10. Deployment Component
class DeploymentComponent(StackComponent):
    """
    Model deployment component.
    """
    
    def initialize(self):
        """Initialize deployment."""
        self.deployment_type = self.config.get('type', 'bentoml')
        
    def deploy_model(self, model, name, version=None):
        """Deploy a model."""
        if self.deployment_type == 'bentoml':
            try:
                import bentoml
                bentoml.save(model, name)
                # More complex deployment would happen here
                return f"Model {name} saved to BentoML repository"
            except ImportError:
                print("Warning: BentoML not installed. Deployment disabled.")
                return None
                
        elif self.deployment_type == 'tf-serving':
            try:
                import tensorflow as tf
                # Save the model in SavedModel format
                export_path = f"./exported_models/{name}/{version or '1'}"
                tf.saved_model.save(model, export_path)
                return f"Model {name} saved for TF Serving at {export_path}"
            except ImportError:
                print("Warning: TensorFlow not installed. Deployment disabled.")
                return None


# 11. File-based datasets
class CSVDataset(Dataset):
    """Dataset for CSV files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        """Load data from CSV."""
        import pandas as pd
        return pd.read_csv(self.filepath)
        
    def save(self, data):
        """Save data to CSV."""
        import os
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data.to_csv(self.filepath, index=False)


class ParquetDataset(Dataset):
    """Dataset for Parquet files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        """Load data from Parquet."""
        import pandas as pd
        return pd.read_parquet(self.filepath)
        
    def save(self, data):
        """Save data to Parquet."""
        import os
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data.to_parquet(self.filepath, index=False)


class PickleDataset(Dataset):
    """Dataset for Pickle files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        """Load data from Pickle."""
        import pickle
        with open(self.filepath, 'rb') as f:
            return pickle.load(f)
            
    def save(self, data):
        """Save data to Pickle."""
        import os
        import pickle
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)


# 12. Helper utilities
def pipeline_from_nodes(*nodes, name=None, tags=None):
    """Create a pipeline from a list of nodes."""
    return Pipeline(list(nodes), name=name, tags=tags or set())


def node_from_func(func, inputs=None, outputs=None, name=None, tags=None):
    """Create a node from a function."""
    return Node(func, inputs, outputs, name, tags)


# 13. CLI Example
def create_cli():
    """Create a CLI for the framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="Pipeline to run")
    run_parser.add_argument("--env", default="local", help="Environment to run in")
    
    # List pipelines command
    list_parser = subparsers.add_parser("list", help="List available pipelines")
    
    # Create project command
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("name", help="Project name")
    
    return parser


# 14. Dummy context manager for cases where MLflow is not available
class DummyContext:
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 15. Example of building a simple pipeline
def example_pipeline():
    """Example of how to build a pipeline."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Define pipeline steps as functions
    def load_data(data_path: str) -> pd.DataFrame:
        """Load raw data."""
        return pd.read_csv(data_path)
    
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        # Drop missing values
        df = df.dropna()
        
        # Convert categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
            
        return df
    
    def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets."""
        target = 'target'  # Assuming 'target' is the target column
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
        """Train a random forest model."""
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(model, X_test, y_test):
        """Evaluate the model."""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}
    
    # Create nodes
    load_node = node_from_func(
        load_data,
        inputs="data_path",
        outputs="raw_data",
        name="load_data",
        tags={"data", "ingestion"}
    )
    
    preprocess_node = node_from_func(
        preprocess_data,
        inputs="raw_data",
        outputs="processed_data",
        name="preprocess_data",
        tags={"data", "preprocessing"}
    )
    
    split_node = node_from_func(
        split_data,
        inputs={"df": "processed_data", "test_size": "params:test_size"},
        outputs=["X_train", "X_test", "y_train", "y_test"],
        name="split_data",
        tags={"data", "splitting"}
    )
    
    train_node = node_from_func(
        train_model,
        inputs={
            "X_train": "X_train",
            "y_train": "y_train",
            "n_estimators": "params:n_estimators"
        },
        outputs="model",
        name="train_model",
        tags={"model", "training"}
    )
    
    evaluate_node = node_from_func(
        evaluate_model,
        inputs={"model": "model", "X_test": "X_test", "y_test": "y_test"},
        outputs="metrics",
        name="evaluate_model",
        tags={"model", "evaluation"}
    )
    
    # Create pipeline
    pipeline = pipeline_from_nodes(
        load_node,
        preprocess_node,
        split_node,
        train_node,
        evaluate_node,
        name="example_pipeline"
    )
    
    return pipeline


# 16. Example of running a pipeline with MLOps stack
def run_example():
    """Run the example pipeline with an MLOps stack."""
    # Create MLOps stack
    stack = Stack()
    
    # Add experiment tracking
    tracker = ExperimentTracker({
        'tracking_uri': 'sqlite:///mlflow.db',
        'experiment_name': 'example'
    })
    stack.add_component('tracking', tracker)
    
    # Add model registry
    registry = ModelRegistry({})
    stack.add_component('registry', registry)
    
    # Add storage
    storage = StorageComponent({'type': 'local'})
    stack.add_component('storage', storage)
    
    # Add deployment
    deployment = DeploymentComponent({'type': 'bentoml'})
    stack.add_component('deployment', deployment)
    
    # Create data catalog
    catalog = DataCatalog()
    catalog.add('data_path', CSVDataset('data/01_raw/data.csv'))
    catalog.add('raw_data', ParquetDataset('data/02_intermediate/raw_data.parquet'))
    catalog.add('processed_data', ParquetDataset('data/03_primary/processed_data.parquet'))
    catalog.add('model', PickleDataset('data/06_models/model.pkl'))
    
    # Create context with parameters
    context = Context(
        catalog=catalog,
        stack=stack,
        params={
            'test_size': 0.2,
            'n_estimators': 100,
            'random_state': 42
        }
    )
    
    # Get pipeline
    pipeline = example_pipeline()
    
    # Run pipeline
    results = context.run(pipeline)
    
    # Deploy model
    model = results.get('model')
    if model:
        stack.deployment.deploy_model(model, 'example_model', version='1.0.0')
        
    return results


# Example pipeline visualization function
def visualize_pipeline(pipeline):
    """Create a visualization of a pipeline."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in pipeline.nodes:
            G.add_node(node.name)
            
        # Add edges
        for node in pipeline.nodes:
            for input_name in node.inputs.values():
                for other_node in pipeline.nodes:
                    if input_name in other_node.outputs.values():
                        G.add_edge(other_node.name, node.name, label=input_name)
        
        # Draw the graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Pipeline Visualization")
        plt.savefig("pipeline_graph.png")
        plt.close()
        
        return "pipeline_graph.png"
    except ImportError:
        print("NetworkX and/or matplotlib not installed. Visualization skipped.")
        return None


# Main function to showcase the framework
if __name__ == "__main__":
    print("ML Framework Example")
    results = run_example()
    print("Pipeline execution results:", results)
