# nexusml/steps/base_step.py
import inspect
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Type, get_type_hints

from nexusml.core.artifact import Artifact
from nexusml.core.context import ExecutionContext

class StepMetadata:
    """Metadata for a step."""
    
    def __init__(self, name: str, inputs: Dict[str, Type], outputs: Dict[str, Type], parameters: Dict[str, Any]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.parameters = parameters
        self.created_at = datetime.now().isoformat()
        self.step_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the step."""
        metadata_str = json.dumps({
            "name": self.name,
            "inputs": str(self.inputs),
            "outputs": str(self.outputs),
            "parameters": str(self.parameters),
            "created_at": self.created_at
        })
        return f"{self.name}-{hashlib.md5(metadata_str.encode()).hexdigest()[:10]}"
    
    def to_dict(self) -> Dict:
        """Convert the metadata to a dictionary."""
        return {
            "name": self.name,
            "step_id": self.step_id,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "parameters": self.parameters,
            "created_at": self.created_at
        }

class BaseStep:
    """Base class for all steps in NexusML."""
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.parameters = kwargs
        self.inputs = {}
        self.outputs = {}
        self._extract_signature()
        self.metadata = StepMetadata(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            parameters=self.parameters
        )
    
    def _extract_signature(self):
        """Extract the step signature from the execute method."""
        sig = inspect.signature(self.execute)
        type_hints = get_type_hints(self.execute)
        
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "context":
                continue
            
            # Check if parameter has a default value
            if param.default is not inspect.Parameter.empty:
                self.parameters.setdefault(param_name, param.default)
            
            # Get the type hint
            if param_name in type_hints:
                param_type = type_hints[param_name]
                self.inputs[param_name] = param_type
        
        # Get the return type hint
        if "return" in type_hints:
            return_type = type_hints["return"]
            # If return type is a dictionary, extract its items
            if hasattr(return_type, "__origin__") and return_type.__origin__ is dict:
                key_type, value_type = return_type.__args__
                if key_type is str:
                    # We assume the return type is Dict[str, Any],
                    # in which case we don't have specific type information
                    # for individual outputs
                    self.outputs = {"output": return_type}
            else:
                self.outputs = {"output": return_type}
    
    def execute(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        """Execute the step."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def __call__(self, context: ExecutionContext, **kwargs) -> Dict[str, Any]:
        """Call the step."""
        # Merge the parameters with the kwargs
        params = {**self.parameters, **kwargs}
        
        # Execute the step
        outputs = self.execute(context, **params)
        
        # Save the outputs to the context
        context.save_step_output(self.metadata.step_id, outputs)
        
        return outputs


# nexusml/steps/decorators.py
from typing import Any, Dict, List, Optional, Callable, Type, get_type_hints, TypeVar, Generic
import inspect

T = TypeVar('T')

class Input(Generic[T]):
    """Decorator for step inputs."""
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        self.name = name
        self.description = description
    
    def __class_getitem__(cls, item):
        """Support for Input[type]."""
        return cls()

class Output(Generic[T]):
    """Decorator for step outputs."""
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        self.name = name
        self.description = description
    
    def __class_getitem__(cls, item):
        """Support for Output[type]."""
        return cls()

class Parameter:
    """Decorator for step parameters."""
    
    def __init__(self, default: Any = None, description: Optional[str] = None):
        self.default = default
        self.description = description

def step(name: Optional[str] = None, description: Optional[str] = None, cache: bool = True):
    """Decorator for defining a step."""
    
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Extract inputs, outputs, and parameters
        inputs = {}
        outputs = {}
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "context":
                continue
            
            # Check if parameter has a default value
            if param.default is not inspect.Parameter.empty:
                parameters[param_name] = param.default
            
            # Get the type hint
            if param_name in type_hints:
                param_type = type_hints[param_name]
                inputs[param_name] = param_type
        
        # Get the return type hint
        if "return" in type_hints:
            return_type = type_hints["return"]
            # If return type is a dictionary, extract its items
            if hasattr(return_type, "__origin__") and return_type.__origin__ is dict:
                key_type, value_type = return_type.__args__
                if key_type is str:
                    # We assume the return type is Dict[str, Any],
                    # in which case we don't have specific type information
                    # for individual outputs
                    outputs = {"output": return_type}
            else:
                outputs = {"output": return_type}
        
        # Create a Step class dynamically
        class StepClass(BaseStep):
            def __init__(self, **kwargs):
                self.name = name or func.__name__
                self.description = description
                self.cache = cache
                super().__init__(name=self.name, **kwargs)
            
            def execute(self, context, **kwargs):
                return func(context=context, **kwargs)
        
        # Return an instance of the step class
        return StepClass
    
    return decorator


# nexusml/pipelines/pipeline.py
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple

from nexusml.core.context import ExecutionContext
from nexusml.steps.base_step import BaseStep

class Pipeline:
    """A pipeline in NexusML."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description
        self.steps = []
        self.created_at = datetime.now().isoformat()
        self.pipeline_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the pipeline."""
        metadata_str = json.dumps({
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at
        })
        return f"{self.name}-{hashlib.md5(metadata_str.encode()).hexdigest()[:10]}"
    
    def add_step(self, step: BaseStep, inputs: Optional[Dict[str, Tuple[str, str]]] = None) -> 'Pipeline':
        """
        Add a step to the pipeline.
        
        Args:
            step: The step to add.
            inputs: Dictionary mapping step input names to (step_name, output_name) tuples.
        """
        self.steps.append((step, inputs or {}))
        return self
    
    def run(self, working_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Args:
            working_dir: Directory to store pipeline artifacts.
            **kwargs: Additional arguments to pass to the first step.
        
        Returns:
            Dict of final outputs.
        """
        # Create a run ID
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create an execution context
        context = ExecutionContext(
            pipeline_id=self.pipeline_id,
            run_id=run_id,
            working_dir=working_dir
        )
        
        # Execute each step
        outputs = {}
        step_outputs = {}
        
        for i, (step, step_inputs) in enumerate(self.steps):
            # Resolve inputs from previous steps
            resolved_inputs = {}
            for input_name, input_source in step_inputs.items():
                source_step_name, output_name = input_source
                if source_step_name not in step_outputs:
                    raise ValueError(f"Step {source_step_name} not found or not executed yet")
                resolved_inputs[input_name] = step_outputs[source_step_name][output_name]
            
            # Add additional inputs from kwargs (for the first step)
            if i == 0:
                resolved_inputs.update(kwargs)
            
            # Execute the step
            step_output = step(context, **resolved_inputs)
            step_outputs[step.name] = step_output
            
            # If this is the last step, store the outputs
            if i == len(self.steps) - 1:
                outputs = step_output
        
        return outputs


# nexusml/pipelines/executor.py
from typing import Dict, List, Optional, Any, Callable, Union
import os
import json
from datetime import datetime

from nexusml.core.context import ExecutionContext
from nexusml.pipelines.pipeline import Pipeline
from nexusml.core.metadata import MetadataStore

class PipelineExecutor:
    """Executor for pipelines."""
    
    def __init__(self, metadata_store: MetadataStore, working_dir: str):
        self.metadata_store = metadata_store
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
    
    def execute(self, pipeline: Pipeline, **kwargs) -> Dict[str, Any]:
        """Execute a pipeline."""
        # Create a run directory
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = os.path.join(self.working_dir, pipeline.name, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save pipeline metadata
        self.metadata_store.save_pipeline_metadata(
            pipeline_id=pipeline.pipeline_id,
            name=pipeline.name,
            created_at=pipeline.created_at,
            metadata={"description": pipeline.description}
        )
        
        # Execute the pipeline
        outputs = pipeline.run(working_dir=run_dir, **kwargs)
        
        # Save the outputs
        outputs_path = os.path.join(run_dir, "outputs.json")
        with open(outputs_path, "w") as f:
            # We can't directly serialize the outputs, so just save the structure
            serializable_outputs = {
                key: f"<{type(value).__name__}>"
                for key, value in outputs.items()
            }
            json.dump(serializable_outputs, f, indent=2)
        
        return outputs
    
    def get_pipeline_runs(self, pipeline_name: str) -> List[str]:
        """Get all runs for a pipeline."""
        pipeline_dir = os.path.join(self.working_dir, pipeline_name)
        if not os.path.exists(pipeline_dir):
            return []
        
        return [
            run_id for run_id in os.listdir(pipeline_dir)
            if os.path.isdir(os.path.join(pipeline_dir, run_id))
        ]
    
    def get_run_outputs(self, pipeline_name: str, run_id: str) -> Dict[str, Any]:
        """Get the outputs of a pipeline run."""
        outputs_path = os.path.join(self.working_dir, pipeline_name, run_id, "outputs.json")
        if not os.path.exists(outputs_path):
            return {}
        
        with open(outputs_path, "r") as f:
            return json.load(f)
