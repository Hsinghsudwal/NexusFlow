from typing import Callable, List, Optional, Dict, Any


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
