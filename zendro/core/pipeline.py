from typing import List, Dict, Optional, Union
from .node import Node


class Pipeline:
    """A pipeline is a directed acyclic graph of nodes."""
    
    def __init__(self, nodes: List[Node], name: Optional[str] = None):
        self.nodes = nodes
        self.name = name or f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._validate()
        
    def _validate(self):
        """Validate pipeline structure for circular dependencies and missing inputs."""
        all_inputs = set()
        all_outputs = set()
        
        for node in self.nodes:
            all_inputs.update(node.inputs)
            all_outputs.update(node.outputs)
            
            output_counts = {}
            for output in node.outputs:
                output_counts[output] = output_counts.get(output, 0) + 1
                
            duplicates = [name for name, count in output_counts.items() if count > 1]
            if duplicates:
                raise ValueError(f"Duplicate outputs in node '{node.name}': {duplicates}")
        
        output_producers = {}
        for node in self.nodes:
            for output in node.outputs:
                if output in output_producers:
                    raise ValueError(f"Output '{output}' is produced by multiple nodes: '{output_producers[output]}' and '{node.name}'")
                output_producers[output] = node.name
        
        missing_inputs = all_inputs - all_outputs
        if missing_inputs:
            print(f"Pipeline has external inputs: {missing_inputs}")
    
    def run(self, catalog, from_nodes: Optional[List[str]] = None, to_nodes: Optional[List[str]] = None, only_nodes: Optional[List[str]] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute the pipeline nodes in dependency order."""
        nodes_to_run = self.nodes
        results = {}
        start_time = datetime.datetime.now()
        print(f"Starting pipeline '{self.name}'")
        
        try:
            for node in self.nodes:
                node_results = node.run(catalog)
                results.update(node_results)
                
            duration = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Pipeline '{self.name}' completed in {duration:.2f}s")
            return results
            
        except Exception as e:
            print(f"Pipeline '{self.name}' failed at node '{node.name}': {str(e)}")
            raise
