

from typing import Any, Dict, List, Callable, Optional, TypeVar, Generic
import inspect
import networkx as nx
from pydantic import validate_arguments

class ArtifactStore:
    """
    Stores and retrieves artifacts produced during pipeline execution.
    """
    def __init__(self, storage_path: str = "./artifacts"):
        self.storage_path = storage_path
        self.cache = {}
    
    def store(self, key: str, data: Any) -> str:
        """Store an artifact and return its URI"""
        self.cache[key] = data
        return key
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve an artifact by key"""
        return self.cache.get(key)
    
    def load_artifact(self, artifact_uri: str) -> Any:
        """Load an artifact by URI"""
        return self.get(artifact_uri)


class MetadataTracker:
    """
    Tracks metadata for pipeline execution.
    """
    def __init__(self):
        self.step_logs = []
    
    def log_step(self, step_name: str, inputs: Any, outputs: Any, artifact_uri: str):
        """Log information about a step execution"""
        self.step_logs.append({
            "step_name": step_name,
            "inputs": inputs,
            "outputs": outputs,
            "artifact_uri": artifact_uri
        })


class StepOutput:
    """
    Container for step execution results.
    """
    def __init__(self, value: Any, artifact_uri: str):
        self.value = value
        self.artifact_uri = artifact_uri


class Step:
    """
    A single execution unit in a pipeline.
    """
    def __init__(
        self,
        func: Callable,
        inputs: Optional[Any] = None,
        outputs: Optional[Any] = None,
        enable_cache: bool = True
    ):
        self.func = func
        self.name = func.__name__
        self.enable_cache = enable_cache
        self.inputs = inputs
        self.outputs = outputs
        self._validate_signature()
    
    def _validate_signature(self):
        """Validate function signature against input/output specifications"""
        # Implementation would check that function signature is compatible with inputs/outputs
        pass
    
    def _generate_cache_key(self, input_data: Any) -> str:
        """Generate a cache key based on input data"""
        # Simple implementation - in practice would use hashing
        import hashlib
        import json
        
        if isinstance(input_data, dict):
            serializable = {k: str(v) for k, v in input_data.items()}
        else:
            serializable = str(input_data)
            
        return f"{self.name}_{hashlib.md5(json.dumps(serializable).encode()).hexdigest()}"
    
    @validate_arguments
    def execute(self, input_data: Any, context: Dict) -> StepOutput:
        """Execute the step function with given inputs and context"""
        # Generate cache key
        cache_key = self._generate_cache_key(input_data)
        
        # Check cache
        if self.enable_cache:
            cached = context["artifact_store"].get(cache_key)
            if cached:
                return StepOutput(value=cached, artifact_uri=cache_key)

        # Execute function
        result = self.func(input_data)
        
        # Validate output (simplified)
        if self.outputs:
            # In practice, would validate against output schema
            pass

        # Store artifact
        artifact_uri = context["artifact_store"].store(
            key=cache_key,
            data=result
        )
        
        # Track metadata
        context["metadata_tracker"].log_step(
            step_name=self.name,
            inputs=input_data,
            outputs=result,
            artifact_uri=artifact_uri
        )

        return StepOutput(value=result, artifact_uri=artifact_uri)


def step(
    enable_cache: bool = True,
    inputs: Optional[Any] = None,
    outputs: Optional[Any] = None
):
    """Decorator for creating pipeline steps"""
    def decorator(func: Callable):
        return Step(
            func=func,
            enable_cache=enable_cache,
            inputs=inputs,
            outputs=outputs
        )
    return decorator


class Orchestrator:
    """
    Orchestrates the execution of pipelines.
    """
    TYPE = "orchestrator"
    
    def __init__(self):
        pass
    
    def create_execution_plan(self, dag: nx.DiGraph) -> List[str]:
        """Create an execution plan based on the DAG"""
        return list(nx.topological_sort(dag))
    
    def execute(self, execution_plan: List[str], initial_inputs: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Execute a pipeline according to the execution plan"""
        results = initial_inputs or {}
        
        for step_name in execution_plan:
            step = context["pipeline"].steps[step_name]
            
            # Collect inputs for this step
            step_inputs = {k: v for k, v in results.items() if k in context["pipeline"].dag.predecessors(step_name)}
            
            # Execute step
            output = step.execute(step_inputs, context)
            
            # Store result
            results[step_name] = output.value
            
        return results
    
    def execute_pipeline(self, pipeline, stack):
        """Execute a complete pipeline"""
        context = {
            "artifact_store": stack.get_component(ArtifactStore.TYPE),
            "metadata_tracker": stack.get_component(MetadataTracker.TYPE),
            "pipeline": pipeline
        }
        
        execution_plan = self.create_execution_plan(pipeline.dag)
        return self.execute(execution_plan, {}, context)


class Stack:
    """
    Container for pipeline execution components.
    """
    _active_stack = None
    
    def __init__(self, name: str):
        self.name = name
        self.components = {}
    
    def add_component(self, component_type: str, component):
        """Add a component to the stack"""
        self.components[component_type] = component
        return self
    
    def get_component(self, component_type: str):
        """Get a component from the stack"""
        return self.components.get(component_type)
    
    def validate(self):
        """Validate that all required components are present"""
        required_components = [ArtifactStore.TYPE, MetadataTracker.TYPE, Orchestrator.TYPE]
        for component in required_components:
            if component not in self.components:
                raise ValueError(f"Stack is missing required component: {component}")
    
    @classmethod
    def get_active_stack(cls):
        """Get the currently active stack"""
        if cls._active_stack is None:
            raise ValueError("No active stack found")
        return cls._active_stack
    
    @classmethod
    def set_active_stack(cls, stack):
        """Set the active stack"""
        cls._active_stack = stack




class Pipeline:
    """
    A pipeline for orchestrating a series of processing steps.
    """
    def __init__(
        self,
        name: str,
        steps: List[Step] = None,
        dependencies: Dict[str, List[str]] = None
    ):
        self.name = name
        self.steps = {step.name: step for step in (steps or [])}
        self.dag = nx.DiGraph()
        
        if dependencies:
            self._build_dag(dependencies)
    
    def step(self, func=None, **kwargs):
        """Decorator to add a step to the pipeline"""
        if func is None:
            return lambda f: self.step(f, **kwargs)
        
        step_obj = Step(func, **kwargs)
        self.steps[step_obj.name] = step_obj
        self.dag.add_node(step_obj.name)
        return func
    
    def _build_dag(self, dependencies: Dict[str, List[str]]) -> nx.DiGraph:
        """Build a directed acyclic graph representing the pipeline"""
        # Add all nodes
        for step_name in self.steps:
            self.dag.add_node(step_name)
        
        # Add edges based on dependencies
        for source, targets in dependencies.items():
            for target in targets:
                self.dag.add_edge(source, target)
        
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Invalid pipeline dependencies - cycle detected")
        
        return self.dag
    
    def add_dependency(self, source: str, target: str):
        """Add a dependency between steps"""
        if source not in self.steps:
            raise ValueError(f"Source step '{source}' not found in pipeline")
        if target not in self.steps:
            raise ValueError(f"Target step '{target}' not found in pipeline")
            
        self.dag.add_edge(source, target)
        
        if not nx.is_directed_acyclic_graph(self.dag):
            self.dag.remove_edge(source, target)
            raise ValueError(f"Adding dependency from {source} to {target} would create a cycle")
    
    def run(self, initial_inputs: Dict[str, Any] = None):
        """Run the pipeline with optional initial inputs"""
        stack = Stack.get_active_stack()
        stack.validate()
        
        orchestrator = stack.get_component(Orchestrator.TYPE)
        context = {
            "artifact_store": stack.get_component(ArtifactStore.TYPE),
            "metadata_tracker": stack.get_component(MetadataTracker.TYPE),
            "pipeline": self
        }
        
        execution_plan = orchestrator.create_execution_plan(self.dag)
        return orchestrator.execute(execution_plan, initial_inputs or {}, context)
    
    def visualize(self):
        """Visualize the pipeline as a graph"""
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.dag)
        nx.draw(self.dag, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        plt.title(f"Pipeline: {self.name}")
        plt.show()


# Example usage
def create_default_stack(name: str = "default"):
    """Create a default stack with standard components"""
    stack = Stack(name)
    stack.add_component(ArtifactStore.TYPE, ArtifactStore())
    stack.add_component(MetadataTracker.TYPE, MetadataTracker())
    stack.add_component(Orchestrator.TYPE, Orchestrator())
    Stack.set_active_stack(stack)
    return stack






class Pipeline:
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps
        self.stack = Stack.get_active_stack()

    def run(self):
        orchestrator = self.stack.orchestrator
        return orchestrator.execute(self.steps)

# -----------------------------
# Pipeline & Steps
# -----------------------------

class Step:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def execute(self, context: Dict[str, Any], artifact_store: ArtifactStore):
        # Resolve inputs from context
        kwargs = {
            k: artifact_store.load_artifact(v) 
            for k, v in context.items() 
            if k in inspect.signature(self.func).parameters
        }
        return self.func(**kwargs)

class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps: List[Step] = []

    def step(self, func):
        self.steps.append(Step(func))
        return func

    def run(self, stack: Stack):
        stack.validate()
        orchestrator = stack.get_component(Orchestrator.TYPE)
        orchestrator.execute_pipeline(self, stack)





class Step:
    def __init__(
        self,
        func: Callable,
        inputs: = None,
        outputs:= None,
        enable_cache: bool = True
    ):
        self.func = func
        self.enable_cache = enable_cache
        self.inputs = inputs
        self.outputs = outputs
        self._validate_signature()


    @validate_arguments
    def execute(self, input_data: Any, context: Dict) -> StepOutput:
        # Generate cache key
        cache_key = self._generate_cache_key(input_data)
        
        # Check cache
        if self.enable_cache:
            cached = context["artifact_store"].get(cache_key)
            if cached:
                return StepOutput(value=cached, artifact_uri=cache_key)

        # Execute function
        result = self.func(input_data)
        
        # Validate output
        if self.output_model:
            result = self.output_model(**result).dict()

        # Store artifact
        artifact_uri = context["artifact_store"].store(
            key=cache_key,
            data=result
        )
        
        # Track metadata
        context["metadata_tracker"].log_step(
            step_name=self.func.__name__,
            inputs=input_data,
            outputs=result,
            artifact_uri=artifact_uri
        )

        return StepOutput(value=result, artifact_uri=artifact_uri)


def step(
    enable_cache: bool = True,
    inputs= None,
    outputs = None
):
    def decorator(func: Callable):
        return Step(
            func=func,
            enable_cache=enable_cache,
            input_model=input_model,
            output_model=output_model
        )
    return decorator



class Pipeline:
    def __init__(
        self,
        name: str,
        steps: List[Step],
        dependencies: Dict[str, List[str]] = None
    ):
        self.name = name
        self.steps = {step.func.__name__: step for step in steps}
        self.dag = self._build_dag(dependencies or {})

 


