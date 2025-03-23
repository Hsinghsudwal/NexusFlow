from typing import Dict, List, Callable, Any, Optional, Union
import concurrent.futures
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class Node:
    """A node in the pipeline that performs a specific task."""
    id: str
    func: Callable
    name: str = ""
    description: str = ""
    outputs: List[Any] = field(default_factory=list)
    next_nodes: List['Node'] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.name:
            self.name = self.id
    
    def add_next(self, node: 'Node') -> 'Node':
        """Connect this node to a next node in the pipeline."""
        self.next_nodes.append(node)
        return self
    
    def execute(self, input_data: Any = None) -> Any:
        """Execute the node's function and store its output."""
        result = self.func(input_data)
        self.outputs.append(result)
        return result


class Pipeline:
    """A pipeline that orchestrates the execution of nodes."""
    def __init__(self, name: str, description: str = "", execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        self.name = name
        self.description = description
        self.execution_mode = execution_mode
        self.nodes: Dict[str, Node] = {}
        self.start_nodes: List[Node] = []
        self.results: Dict[str, Any] = {}
        self.execution_times: Dict[str, float] = {}
    
    def add_node(self, node: Node) -> Node:
        """Add a node to the pipeline."""
        self.nodes[node.id] = node
        return node
    
    def set_start_node(self, node: Union[Node, str]) -> None:
        """Set a node as a start node for the pipeline."""
        if isinstance(node, str):
            if node in self.nodes:
                self.start_nodes.append(self.nodes[node])
            else:
                raise ValueError(f"Node with ID {node} not found in pipeline")
        else:
            if node.id in self.nodes:
                self.start_nodes.append(node)
            else:
                raise ValueError(f"Node {node.id} not added to pipeline yet")
    
    def _execute_sequential(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the pipeline sequentially."""
        if not self.start_nodes:
            raise ValueError("No start nodes defined for the pipeline")
        
        queue = [(node, input_data) for node in self.start_nodes]
        while queue:
            current_node, data = queue.pop(0)
            start_time = time.time()
            result = current_node.execute(data)
            end_time = time.time()
            self.execution_times[current_node.id] = end_time - start_time
            self.results[current_node.id] = result
            
            # Enqueue next nodes
            for next_node in current_node.next_nodes:
                queue.append((next_node, result))
        
        return self.results
    
    def _execute_node_parallel(self, node: Node, input_data: Any) -> None:
        """Execute a single node in parallel mode."""
        start_time = time.time()
        result = node.execute(input_data)
        end_time = time.time()
        
        self.execution_times[node.id] = end_time - start_time
        self.results[node.id] = result
        
        # Process next nodes in parallel
        if node.next_nodes:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._execute_node_parallel, next_node, result)
                    for next_node in node.next_nodes
                ]
                concurrent.futures.wait(futures)
    
    def _execute_parallel(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the pipeline in parallel."""
        if not self.start_nodes:
            raise ValueError("No start nodes defined for the pipeline")
        
        # Execute start nodes in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._execute_node_parallel, node, input_data)
                for node in self.start_nodes
            ]
            concurrent.futures.wait(futures)
        
        return self.results
    
    def run(self, input_data: Any = None) -> Dict[str, Any]:
        """Run the pipeline with the specified execution mode."""
        self.results = {}
        self.execution_times = {}
        
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential(input_data)
        else:
            return self._execute_parallel(input_data)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about the last pipeline execution."""
        if not self.execution_times:
            return {"error": "Pipeline has not been executed yet"}
        
        total_time = sum(self.execution_times.values())
        return {
            "total_execution_time": total_time,
            "node_times": self.execution_times,
            "node_count": len(self.nodes),
            "results": {k: type(v).__name__ for k, v in self.results.items()}
        }


def create_node(func: Callable, name: str = "", description: str = "", node_id: str = None) -> Node:
    """Helper function to create a node."""
    if node_id is None:
        node_id = f"node_{uuid.uuid4().hex[:8]}"
    return Node(id=node_id, func=func, name=name, description=description)
