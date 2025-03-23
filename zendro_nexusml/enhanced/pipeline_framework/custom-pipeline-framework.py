import os
import pickle
import yaml
import logging
import datetime
import importlib
import inspect
import hashlib
from typing import Any, Dict, List, Callable, Optional, Union, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataCatalog:
    """Data catalog for managing dataset storage and retrieval."""
    base_path: Path

    def __post_init__(self):
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Save data to the catalog."""
        data_path = self.base_path / f"{name}.pkl"
        metadata_path = self.base_path / f"{name}.meta.yaml"
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata if provided
        if metadata:
            current_metadata = {
                'created_at': datetime.datetime.now().isoformat(),
                'data_hash': hashlib.md5(pickle.dumps(data)).hexdigest()
            }
            current_metadata.update(metadata)
            
            with open(metadata_path, 'w') as f:
                yaml.dump(current_metadata, f)
                
        logger.info(f"Saved data '{name}' to {data_path}")

    def load(self, name: str) -> Any:
        """Load data from the catalog."""
        data_path = self.base_path / f"{name}.pkl"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data '{name}' not found in catalog")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Loaded data '{name}' from {data_path}")
        return data
    
    def exists(self, name: str) -> bool:
        """Check if data exists in the catalog."""
        data_path = self.base_path / f"{name}.pkl"
        return data_path.exists()
    
    def get_metadata(self, name: str) -> Dict:
        """Get metadata for a dataset."""
        metadata_path = self.base_path / f"{name}.meta.yaml"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata for '{name}' not found in catalog")
        
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)

class Node:
    """A node represents a single processing step in a pipeline."""
    
    def __init__(
        self, 
        func: Callable, 
        inputs: List[str], 
        outputs: List[str],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        self.func = func
        self.inputs = inputs
        self.outputs = outputs
        self.name = name or func.__name__
        self.tags = set(tags or [])
        
    def run(self, catalog: DataCatalog) -> Dict[str, Any]:
        """Execute the node function with inputs from catalog and store outputs."""
        # Load inputs from catalog
        input_data = {}
        for input_name in self.inputs:
            input_data[input_name] = catalog.load(input_name)
        
        # Execute function
        logger.info(f"Running node '{self.name}'")
        if self.inputs:
            outputs = self.func(**input_data)
        else:
            outputs = self.func()
        
        # Handle different output structures
        if len(self.outputs) == 1:
            outputs = {self.outputs[0]: outputs}
        elif not isinstance(outputs, tuple) and len(self.outputs) > 1:
            raise ValueError(f"Expected {len(self.outputs)} outputs but got a single value")
        elif isinstance(outputs, tuple) and len(outputs) != len(self.outputs):
            raise ValueError(f"Expected {len(self.outputs)} outputs but got {len(outputs)}")
        else:
            outputs = dict(zip(self.outputs, outputs))
        
        # Save outputs to catalog
        for name, data in outputs.items():
            catalog.save(name, data, metadata={'node': self.name})
            
        return outputs
    
    def __repr__(self):
        return f"Node(name='{self.name}', inputs={self.inputs}, outputs={self.outputs}, tags={self.tags})"

class Pipeline:
    """A pipeline is a directed acyclic graph of nodes."""
    
    def __init__(
        self, 
        nodes: List[Node],
        name: Optional[str] = None
    ):
        self.nodes = nodes
        self.name = name or f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._validate()
        
    def _validate(self):
        """Validate pipeline structure for circular dependencies and missing inputs."""
        # Build dependency graph
        all_inputs = set()
        all_outputs = set()
        
        for node in self.nodes:
            all_inputs.update(node.inputs)
            all_outputs.update(node.outputs)
            
            # Check for duplicate outputs
            output_counts = {}
            for output in node.outputs:
                output_counts[output] = output_counts.get(output, 0) + 1
                
            duplicates = [name for name, count in output_counts.items() if count > 1]
            if duplicates:
                raise ValueError(f"Duplicate outputs in node '{node.name}': {duplicates}")
        
        # Check for outputs produced by multiple nodes
        output_producers = {}
        for node in self.nodes:
            for output in node.outputs:
                if output in output_producers:
                    raise ValueError(
                        f"Output '{output}' is produced by multiple nodes: "
                        f"'{output_producers[output]}' and '{node.name}'"
                    )
                output_producers[output] = node.name
        
        # Check for missing inputs (inputs not produced by any node)
        missing_inputs = all_inputs - all_outputs
        if missing_inputs:
            logger.warning(f"Pipeline has external inputs: {missing_inputs}")
            
    def _sort_nodes(self) -> List[Node]:
        """Sort nodes topologically to determine execution order."""
        # Build dependency graph
        dependencies = {node.name: set() for node in self.nodes}
        outputs_to_node = {}
        
        for node in self.nodes:
            for output in node.outputs:
                outputs_to_node[output] = node.name
                
        for node in self.nodes:
            for input_name in node.inputs:
                if input_name in outputs_to_node:
                    dependencies[node.name].add(outputs_to_node[input_name])
        
        # Topological sort
        sorted_nodes = []
        visited = set()
        temp_visited = set()
        
        def visit(node_name):
            if node_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving node '{node_name}'")
            
            if node_name not in visited:
                temp_visited.add(node_name)
                
                for dep in dependencies[node_name]:
                    visit(dep)
                    
                temp_visited.remove(node_name)
                visited.add(node_name)
                sorted_nodes.append(next(node for node in self.nodes if node.name == node_name))
        
        for node in self.nodes:
            if node.name not in visited:
                visit(node.name)
                
        return list(reversed(sorted_nodes))
    
    def run(
        self, 
        catalog: DataCatalog,
        from_nodes: Optional[List[str]] = None,
        to_nodes: Optional[List[str]] = None,
        only_nodes: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute the pipeline nodes in dependency order."""
        # Filter nodes based on parameters
        nodes_to_run = self.nodes
        
        if only_nodes:
            nodes_to_run = [n for n in nodes_to_run if n.name in only_nodes]
        else:
            if from_nodes:
                include_after = False
                filtered_nodes = []
                for node in self._sort_nodes():
                    if node.name in from_nodes:
                        include_after = True
                    if include_after:
                        filtered_nodes.append(node)
                nodes_to_run = filtered_nodes
                
            if to_nodes:
                include_before = True
                filtered_nodes = []
                for node in self._sort_nodes():
                    if include_before:
                        filtered_nodes.append(node)
                    if node.name in to_nodes:
                        include_before = False
                nodes_to_run = filtered_nodes
        
        if tags:
            tag_set = set(tags)
            nodes_to_run = [n for n in nodes_to_run if n.tags.intersection(tag_set)]
        
        # Sort nodes to respect dependencies
        execution_order = []
        all_node_names = {node.name for node in self.nodes}
        for node in self._sort_nodes():
            if node.name in {n.name for n in nodes_to_run}:
                execution_order.append(node)
        
        results = {}
        start_time = datetime.datetime.now()
        logger.info(f"Starting pipeline '{self.name}'")
        
        try:
            for node in execution_order:
                node_start = datetime.datetime.now()
                node_results = node.run(catalog)
                node_duration = (datetime.datetime.now() - node_start).total_seconds()
                logger.info(f"Completed node '{node.name}' in {node_duration:.2f}s")
                results.update(node_results)
                
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline '{self.name}' completed in {duration:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline '{self.name}' failed at node '{node.name}': {str(e)}")
            raise
    
    def __repr__(self):
        return f"Pipeline(name='{self.name}', nodes={len(self.nodes)})"

def node(
    func: Optional[Callable] = None,
    inputs: Optional[List[str]] = None,
    outputs: Optional[Union[str, List[str]]] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Union[Node, Callable]:
    """Decorator to create a node from a function."""
    
    # Handle case when decorator is used without parentheses
    if func is not None:
        return Node(
            func=func,
            inputs=inputs or [],
            outputs=[outputs] if isinstance(outputs, str) else (outputs or []),
            name=name or func.__name__,
            tags=tags or []
        )
    
    # Handle case when decorator is used with parentheses
    def decorator(function):
        return Node(
            func=function,
            inputs=inputs or [],
            outputs=[outputs] if isinstance(outputs, str) else (outputs or []),
            name=name or function.__name__,
            tags=tags or []
        )
    
    return decorator

class ProjectContext:
    """Context object for managing project resources."""
    
    def __init__(self, project_path: Union[str, Path]):
        self.project_path = Path(project_path)
        self.config = self._load_config()
        self.catalog = DataCatalog(self.project_path / self.config.get('data_path', 'data'))
        
    def _load_config(self) -> Dict:
        """Load project configuration."""
        config_path = self.project_path / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def load_pipeline(self, module_name: str, pipeline_name: str = 'pipeline') -> Pipeline:
        """Load a pipeline from a Python module."""
        try:
            module = importlib.import_module(module_name)
            pipeline = getattr(module, pipeline_name)
            if not isinstance(pipeline, Pipeline):
                raise TypeError(f"Object '{pipeline_name}' in module '{module_name}' is not a Pipeline")
            return pipeline
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load pipeline '{pipeline_name}' from module '{module_name}': {str(e)}")
            
    def run_pipeline(
        self, 
        pipeline: Union[Pipeline, str],
        pipeline_name: str = 'pipeline',
        **kwargs
    ) -> Dict[str, Any]:
        """Run a pipeline with the project context."""
        if isinstance(pipeline, str):
            pipeline = self.load_pipeline(pipeline, pipeline_name)
            
        return pipeline.run(self.catalog, **kwargs)

def create_project(project_path: Union[str, Path]) -> None:
    """Create a new project folder structure."""
    project_path = Path(project_path)
    
    if project_path.exists() and any(project_path.iterdir()):
        raise ValueError(f"Directory {project_path} already exists and is not empty")
    
    # Create directories
    (project_path / 'data').mkdir(parents=True, exist_ok=True)
    (project_path / 'src' / 'pipelines').mkdir(parents=True, exist_ok=True)
    (project_path / 'notebooks').mkdir(parents=True, exist_ok=True)
    (project_path / 'conf').mkdir(parents=True, exist_ok=True)
    
    # Create config file
    config = {
        'data_path': 'data',
        'project_name': project_path.name
    }
    
    with open(project_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create example pipeline
    example_pipeline = """import pandas as pd
from datapipe import node, Pipeline

@node(outputs="raw_data")
def load_data():
    # Example function to load data
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

@node(inputs="raw_data", outputs="processed_data")
def process_data(raw_data):
    # Example function to process data
    processed = raw_data.copy()
    processed['C'] = processed['A'] + processed['B']
    return processed

@node(inputs="processed_data", outputs="results")
def analyze_data(processed_data):
    # Example function to analyze data
    result = {
        'mean': processed_data.mean().to_dict(),
        'sum': processed_data.sum().to_dict()
    }
    return result

# Create pipeline
pipeline = Pipeline(
    nodes=[
        load_data,
        process_data,
        analyze_data
    ],
    name="example_pipeline"
)
"""
    
    with open(project_path / 'src' / 'pipelines' / 'example.py', 'w') as f:
        f.write(example_pipeline)
    
    # Create README
    readme = f"""# {project_path.name}

A data pipeline project created with the custom pipeline framework.

## Structure

- `data/`: Data storage directory
- `src/pipelines/`: Pipeline definitions
- `notebooks/`: Jupyter notebooks for exploration
- `conf/`: Configuration files

## Running the example pipeline

```python
from datapipe import ProjectContext

context = ProjectContext("{project_path}")
results = context.run_pipeline("src.pipelines.example")
print(results)
```
"""
    
    with open(project_path / 'README.md', 'w') as f:
        f.write(readme)
        
    print(f"Created new project at {project_path}")
