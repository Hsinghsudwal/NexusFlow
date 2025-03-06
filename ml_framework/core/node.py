# 1. Node 
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

        self.name = name or self.__class__.__name__
        self.uuid = str(uuid.uuid4())
        self._output_artifacts = {}
        self._input_artifacts = {}
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





    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.uuid = str(uuid.uuid4())
        self._output_artifacts = {}
        self._input_artifacts = {}
        
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the node logic."""
        pass
        
    def __call__(self, *args, **kwargs) -> Any:
        """Make the node callable."""
        # This would normally capture inputs and outputs for the metadata store
        # but simplified here
        start_time = datetime.datetime.now()
        result = self.process(*args, **kwargs)
        end_time = datetime.datetime.now()
        
        # Log execution time
        execution_time = (end_time - start_time).total_seconds()
        print(f"Node '{self.name}' executed in {execution_time:.2f} seconds")
        
        return result
        
    def connect(self, **connections):
        """Connect this step to other steps."""
        self._input_artifacts.update(connections)
        return self
        

    def node(func=None, *, name=None, materializers=None):

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