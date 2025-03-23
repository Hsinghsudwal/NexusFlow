import os
import json
import pickle
import datetime
import inspect
import hashlib
from typing import Any, Dict, List, Callable, Optional, Union, Type
from dataclasses import dataclass, field
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml-workflow')

# ===== Artifact Management =====

class Artifact:
    """Base class for all artifacts."""
    
    def __init__(self, data: Any, metadata: Optional[Dict] = None):
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        
    def save(self, path: str) -> None:
        """Save artifact to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Save metadata separately in JSON format
        metadata_path = f"{path}.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                **self.metadata,
                'created_at': self.created_at.isoformat(),
                'type': self.__class__.__name__
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Artifact':
        """Load artifact from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class DatasetArtifact(Artifact):
    """Artifact for datasets."""
    
    def __init__(self, data: Any, metadata: Optional[Dict] = None):
        super().__init__(data, metadata)
        # Add dataset-specific metadata
        if self.metadata is None:
            self.metadata = {}
        if hasattr(data, 'shape'):
            self.metadata['shape'] = data.shape
        

class ModelArtifact(Artifact):
    """Artifact for ML models."""
    
    def __init__(self, model: Any, metadata: Optional[Dict] = None):
        super().__init__(model, metadata)
        # Add model-specific metadata
        if self.metadata is None:
            self.metadata = {}


# ===== Step Definition =====

@dataclass
class StepConfig:
    """Configuration for a Step."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


class Step:
    """Base class for all pipeline steps."""
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.inputs = {}
        self.outputs = {}
        self._func = None
        self._signature = None
        
    def __call__(self, **kwargs):
        """Make the step callable to register input artifacts."""
        self.inputs = kwargs
        return self
    
    def execute(self) -> Dict[str, Artifact]:
        """Execute the step and return outputs."""
        if self._func is None:
            raise NotImplementedError("Step must implement _execute or register a function")
        
        # Unpack inputs to match function signature
        input_args = {}
        for param_name, param in self._signature.parameters.items():
            if param_name in self.inputs:
                # If input is an artifact, pass its data
                if isinstance(self.inputs[param_name], Artifact):
                    input_args[param_name] = self.inputs[param_name].data
                else:
                    input_args[param_name] = self.inputs[param_name]
        
        # Execute function
        logger.info(f"Executing step: {self.name}")
        result = self._func(**input_args)
        
        # Convert result to artifacts
        if isinstance(result, tuple):
            # Multiple outputs
            outputs = {}
            for i, res in enumerate(result):
                output_name = f"output_{i}"
                outputs[output_name] = Artifact(res)
            self.outputs = outputs
        else:
            # Single output
            self.outputs = {"output": Artifact(result)}
        
        return self.outputs
    
    @classmethod
    def from_func(cls, func: Callable, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> 'Step':
        """Create a step from a function."""
        step = cls(name or func.__name__, config)
        step._func = func
        step._signature = inspect.signature(func)
        return step


# ===== Pipeline Definition =====

class Pipeline:
    """Pipeline for orchestrating steps."""
    
    def __init__(self, name: str, artifact_store: Optional[str] = None):
        self.name = name
        self.steps: Dict[str, Step] = {}
        self.step_dependencies: Dict[str, List[str]] = {}
        self.artifact_store = artifact_store or os.path.join('artifacts', name)
        self.artifacts: Dict[str, Dict[str, Artifact]] = {}
        self.execution_order: List[str] = []
        self.executed = False
        
    def add_step(self, step: Step, dependencies: Optional[List[str]] = None) -> 'Pipeline':
        """Add a step to the pipeline."""
        self.steps[step.name] = step
        self.step_dependencies[step.name] = dependencies or []
        return self
    
    def _resolve_execution_order(self) -> List[str]:
        """Resolve step execution order based on dependencies."""
        visited = set()
        temp = set()
        order = []
        
        def visit(step_name):
            if step_name in temp:
                raise ValueError(f"Circular dependency detected for step: {step_name}")
            if step_name in visited:
                return
            
            temp.add(step_name)
            for dep in self.step_dependencies.get(step_name, []):
                visit(dep)
            
            temp.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        return order
    
    def _get_artifact_path(self, step_name: str, output_name: str) -> str:
        """Get path for storing an artifact."""
        pipeline_hash = hashlib.md5(self.name.encode()).hexdigest()[:8]
        return os.path.join(
            self.artifact_store,
            pipeline_hash,
            step_name,
            f"{output_name}.pkl"
        )
    
    def run(self) -> Dict[str, Dict[str, Artifact]]:
        """Run the pipeline."""
        os.makedirs(self.artifact_store, exist_ok=True)
        
        # Resolve execution order
        self.execution_order = self._resolve_execution_order()
        logger.info(f"Pipeline execution order: {self.execution_order}")
        
        # Execute steps
        for step_name in self.execution_order:
            step = self.steps[step_name]
            
            # Connect inputs from previous steps if needed
            for dep_name in self.step_dependencies[step_name]:
                dep_step = self.steps[dep_name]
                for output_name, artifact in dep_step.outputs.items():
                    # If this output is needed as an input for the current step
                    if output_name in step.inputs and step.inputs[output_name] is None:
                        step.inputs[output_name] = artifact
            
            # Execute step
            outputs = step.execute()
            self.artifacts[step_name] = outputs
            
            # Save artifacts
            for output_name, artifact in outputs.items():
                artifact_path = self._get_artifact_path(step_name, output_name)
                artifact.save(artifact_path)
                logger.info(f"Saved artifact: {step_name}.{output_name} to {artifact_path}")
        
        self.executed = True
        return self.artifacts
    
    def visualize(self) -> str:
        """Generate a simple text representation of the pipeline graph."""
        if not self.execution_order:
            self.execution_order = self._resolve_execution_order()
        
        result = [f"Pipeline: {self.name}", ""]
        
        for i, step_name in enumerate(self.execution_order):
            deps = self.step_dependencies[step_name]
            deps_str = f" (depends on: {', '.join(deps)})" if deps else ""
            result.append(f"{i+1}. {step_name}{deps_str}")
        
        return "\n".join(result)


# ===== Framework Core =====

class MLWorkflow:
    """Main entry point for the framework."""
    
    def __init__(self, workspace_dir: str = "ml_workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        self.pipelines: Dict[str, Pipeline] = {}
    
    def pipeline(self, name: str) -> Pipeline:
        """Create or get a pipeline."""
        if name not in self.pipelines:
            artifact_store = os.path.join(self.workspace_dir, "artifacts", name)
            self.pipelines[name] = Pipeline(name, artifact_store)
        return self.pipelines[name]
    
    def step(self, func: Callable = None, *, name: str = None, config: Dict[str, Any] = None) -> Union[Step, Callable]:
        """Decorator to create a step from a function."""
        def decorator(f):
            return Step.from_func(f, name, config)
        
        if func is None:
            return decorator
        return decorator(func)


# ===== Usage Example =====

def example_usage():
    # Initialize the workflow
    workflow = MLWorkflow(workspace_dir="my_ml_project")
    
    # Define steps using the decorator
    @workflow.step
    def load_data(data_path: str):
        # Simulating data loading
        import numpy as np
        data = np.random.rand(100, 10)
        return DatasetArtifact(data, {"source": data_path})
    
    @workflow.step
    def preprocess(dataset):
        # Simulating preprocessing
        processed_data = dataset * 2
        return DatasetArtifact(processed_data, {"preprocessing": "scaled"})
    
    @workflow.step
    def train_model(dataset):
        # Simulating model training
        model = {"weights": [0.1, 0.2, 0.3], "bias": 0.01}
        return ModelArtifact(model, {"training_date": datetime.datetime.now().isoformat()})
    
    @workflow.step
    def evaluate(model, test_data):
        # Simulating evaluation
        score = 0.85
        return {"accuracy": score, "f1": 0.82}
    
    # Create a pipeline
    pipeline = workflow.pipeline("training_pipeline")
    
    # Add steps to the pipeline with dependencies
    pipeline.add_step(load_data(data_path="data/train.csv"))
    pipeline.add_step(preprocess(dataset=None), dependencies=["load_data"])
    pipeline.add_step(train_model(dataset=None), dependencies=["preprocess"])
    pipeline.add_step(evaluate(model=None, test_data=None), dependencies=["train_model", "load_data"])
    
    # Run the pipeline
    results = pipeline.run()
    
    # Print pipeline visualization
    print(pipeline.visualize())
    
    return results


if __name__ == "__main__":
    example_usage()
