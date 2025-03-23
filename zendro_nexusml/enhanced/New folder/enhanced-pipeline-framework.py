from typing import Dict, List, Callable, Any, Optional, Union, Set, Tuple
import concurrent.futures
import time
import uuid
import json
import os
import logging
import traceback
from enum import Enum
from dataclasses import dataclass, field, asdict
import pickle
import functools
import graphviz
from datetime import datetime


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RetryStrategy(Enum):
    NONE = "none"
    SIMPLE = "simple"  # Simple retry with fixed delay
    EXPONENTIAL_BACKOFF = "exponential_backoff"


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class NodeExecutionError(Exception):
    """Exception raised when node execution fails."""
    def __init__(self, node_id: str, original_error: Exception):
        self.node_id = node_id
        self.original_error = original_error
        super().__init__(f"Error executing node {node_id}: {original_error}")


@dataclass
class NodeError:
    """Information about an error that occurred during node execution."""
    timestamp: str
    error_type: str
    error_message: str
    traceback: str
    attempt: int
    

@dataclass
class NodeMetadata:
    """Metadata about a node execution."""
    status: NodeStatus = NodeStatus.PENDING
    start_time: float = 0
    end_time: float = 0
    attempt: int = 0
    errors: List[NodeError] = field(default_factory=list)


@dataclass
class Condition:
    """A condition that determines whether a node should be executed."""
    func: Callable[[Any], bool]
    description: str = ""
    
    def evaluate(self, data: Any) -> bool:
        """Evaluate the condition with the given data."""
        return self.func(data)


@dataclass
class Validator:
    """A validator that checks if data meets certain criteria."""
    func: Callable[[Any], Tuple[bool, str]]
    description: str = ""
    
    def validate(self, data: Any) -> Tuple[bool, str]:
        """Validate the data and return (is_valid, error_message)."""
        return self.func(data)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    strategy: RetryStrategy = RetryStrategy.NONE
    max_attempts: int = 3
    delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds (for exponential backoff)
    backoff_factor: float = 2.0
    
    def get_delay_for_attempt(self, attempt: int) -> float:
        """Calculate the delay for a specific retry attempt."""
        if self.strategy == RetryStrategy.NONE or attempt <= 0:
            return 0
        
        if self.strategy == RetryStrategy.SIMPLE:
            return self.delay
        
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.delay * (self.backoff_factor ** (attempt - 1))
            return min(delay, self.max_delay)


@dataclass
class Node:
    """A node in the pipeline that performs a specific task."""
    id: str
    func: Callable
    name: str = ""
    description: str = ""
    outputs: List[Any] = field(default_factory=list)
    next_nodes: List['Node'] = field(default_factory=list)
    conditional_next_nodes: List[Tuple['Node', Condition]] = field(default_factory=list)
    validators: List[Validator] = field(default_factory=list)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    
    def __post_init__(self):
        if not self.name:
            self.name = self.id
    
    def add_next(self, node: 'Node') -> 'Node':
        """Connect this node to a next node in the pipeline."""
        self.next_nodes.append(node)
        return self
    
    def add_conditional_next(self, node: 'Node', condition: Condition) -> 'Node':
        """Connect this node to a next node with a condition."""
        self.conditional_next_nodes.append((node, condition))
        return self
    
    def add_validator(self, validator: Validator) -> 'Node':
        """Add a data validator to this node."""
        self.validators.append(validator)
        return self
    
    def set_retry_config(self, 
                         strategy: RetryStrategy, 
                         max_attempts: int = 3, 
                         delay: float = 1.0,
                         max_delay: float = 60.0,
                         backoff_factor: float = 2.0) -> 'Node':
        """Set the retry configuration for this node."""
        self.retry_config = RetryConfig(
            strategy=strategy,
            max_attempts=max_attempts,
            delay=delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor
        )
        return self
    
    def _validate_data(self, data: Any) -> None:
        """Validate the input data using all validators."""
        for validator in self.validators:
            is_valid, error_message = validator.validate(data)
            if not is_valid:
                raise DataValidationError(f"Validation failed for node {self.id}: {error_message}")
    
    def execute(self, input_data: Any = None) -> Any:
        """Execute the node's function with retry logic and store its output."""
        self.metadata.status = NodeStatus.RUNNING
        self.metadata.start_time = time.time()
        self.metadata.attempt = 0
        
        result = None
        success = False
        
        try:
            if self.validators:
                self._validate_data(input_data)
            
            # Retry logic
            max_attempts = max(1, self.retry_config.max_attempts)
            for attempt in range(1, max_attempts + 1):
                self.metadata.attempt = attempt
                try:
                    result = self.func(input_data)
                    success = True
                    break
                except Exception as e:
                    error_info = NodeError(
                        timestamp=datetime.now().isoformat(),
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc(),
                        attempt=attempt
                    )
                    self.metadata.errors.append(error_info)
                    
                    if attempt < max_attempts:
                        delay = self.retry_config.get_delay_for_attempt(attempt)
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        raise NodeExecutionError(self.id, e)
            
            if success:
                self.outputs.append(result)
                self.metadata.status = NodeStatus.COMPLETED
            else:
                self.metadata.status = NodeStatus.FAILED
                
        except Exception as e:
            self.metadata.status = NodeStatus.FAILED
            if not isinstance(e, NodeExecutionError):
                error_info = NodeError(
                    timestamp=datetime.now().isoformat(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    attempt=self.metadata.attempt
                )
                self.metadata.errors.append(error_info)
            raise
        finally:
            self.metadata.end_time = time.time()
            
        return result
    
    def to_dict(self) -> Dict:
        """Convert node to a dictionary for serialization."""
        node_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "retry_config": {
                "strategy": self.retry_config.strategy.value,
                "max_attempts": self.retry_config.max_attempts,
                "delay": self.retry_config.delay,
                "max_delay": self.retry_config.max_delay,
                "backoff_factor": self.retry_config.backoff_factor
            },
            "metadata": {
                "status": self.metadata.status.value,
                "start_time": self.metadata.start_time,
                "end_time": self.metadata.end_time,
                "attempt": self.metadata.attempt,
                "errors": [asdict(error) for error in self.metadata.errors]
            },
            "next_nodes": [node.id for node in self.next_nodes],
            "conditional_next_nodes": [(node.id, cond.description) for node, cond in self.conditional_next_nodes]
        }
        return node_dict


class PipelineError(Exception):
    """Exception raised when pipeline execution fails."""
    pass


@dataclass
class PipelineState:
    """State of a pipeline execution that can be saved and restored."""
    pipeline_id: str
    pipeline_name: str
    execution_mode: str
    node_states: Dict[str, Dict]
    results: Dict[str, Any]
    execution_times: Dict[str, float]
    start_time: float
    end_time: float
    status: str = "not_started"  # not_started, running, completed, failed
    
    def save(self, directory: str = "./pipeline_states") -> str:
        """Save pipeline state to a file."""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.pipeline_id}_{timestamp}.json"
        filepath = os.path.join(directory, filename)
        
        # Create a serializable state dictionary
        state_dict = {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "execution_mode": self.execution_mode,
            "node_states": self.node_states,
            "results": {k: str(v) for k, v in self.results.items()},  # Convert results to strings for JSON serialization
            "execution_times": self.execution_times,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        # Also save a binary version with complete objects using pickle
        pickle_filepath = os.path.join(directory, f"{self.pipeline_id}_{timestamp}.pkl")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(self, f)
            
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineState':
        """Load pipeline state from a file."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
                return cls(**state_dict)


class Pipeline:
    """A pipeline that orchestrates the execution of nodes."""
    def __init__(self, name: str, description: str = "", execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        self.id = f"pipeline_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = description
        self.execution_mode = execution_mode
        self.nodes: Dict[str, Node] = {}
        self.start_nodes: List[Node] = []
        self.results: Dict[str, Any] = {}
        self.execution_times: Dict[str, float] = {}
        self.start_time: float = 0
        self.end_time: float = 0
        self.status: str = "not_started"
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the pipeline."""
        logger = logging.getLogger(f"pipeline.{self.id}")
        logger.setLevel(logging.INFO)
        
        # Create handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Also create a file handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(f"logs/pipeline_{self.id}.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
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
    
    def _get_next_nodes(self, node: Node, data: Any) -> List[Node]:
        """Get all next nodes to execute based on regular and conditional connections."""
        next_nodes = list(node.next_nodes)
        
        # Evaluate conditional next nodes
        for next_node, condition in node.conditional_next_nodes:
            if condition.evaluate(data):
                next_nodes.append(next_node)
                self.logger.info(f"Condition '{condition.description}' for node {node.id} -> {next_node.id} evaluated to True")
            else:
                self.logger.info(f"Condition '{condition.description}' for node {node.id} -> {next_node.id} evaluated to False")
        
        return next_nodes
    
    def _execute_sequential(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the pipeline sequentially."""
        if not self.start_nodes:
            raise ValueError("No start nodes defined for the pipeline")
        
        queue = [(node, input_data) for node in self.start_nodes]
        processed_nodes = set()
        
        while queue:
            current_node, data = queue.pop(0)
            
            # Skip already processed nodes (to avoid cycles)
            if current_node.id in processed_nodes:
                continue
                
            processed_nodes.add(current_node.id)
            
            self.logger.info(f"Executing node {current_node.id} ({current_node.name})")
            try:
                result = current_node.execute(data)
                execution_time = current_node.metadata.end_time - current_node.metadata.start_time
                self.execution_times[current_node.id] = execution_time
                self.results[current_node.id] = result
                
                self.logger.info(f"Node {current_node.id} completed in {execution_time:.2f}s")
                
                # Get and enqueue next nodes
                next_nodes = self._get_next_nodes(current_node, result)
                for next_node in next_nodes:
                    queue.append((next_node, result))
                    
            except Exception as e:
                self.logger.error(f"Error executing node {current_node.id}: {str(e)}")
                # If we encounter an error, we'll still try to process other paths in the pipeline
                continue
        
        return self.results
    
    def _execute_node_parallel(self, node: Node, input_data: Any, processed_nodes: Set[str]) -> None:
        """Execute a single node in parallel mode."""
        # Skip already processed nodes
        if node.id in processed_nodes:
            return
            
        processed_nodes.add(node.id)
        
        self.logger.info(f"Executing node {node.id} ({node.name})")
        try:
            result = node.execute(input_data)
            execution_time = node.metadata.end_time - node.metadata.start_time
            
            # These operations need to be thread-safe
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as update_executor:
                update_executor.submit(lambda: self.execution_times.update({node.id: execution_time}))
                update_executor.submit(lambda: self.results.update({node.id: result}))
            
            self.logger.info(f"Node {node.id} completed in {execution_time:.2f}s")
            
            # Get next nodes to execute
            next_nodes = self._get_next_nodes(node, result)
            
            # Process next nodes in parallel
            if next_nodes:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self._execute_node_parallel, next_node, result, processed_nodes)
                        for next_node in next_nodes
                    ]
                    concurrent.futures.wait(futures)
                    
        except Exception as e:
            self.logger.error(f"Error executing node {node.id}: {str(e)}")
    
    def _execute_parallel(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the pipeline in parallel."""
        if not self.start_nodes:
            raise ValueError("No start nodes defined for the pipeline")
        
        processed_nodes = set()
        
        # Execute start nodes in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._execute_node_parallel, node, input_data, processed_nodes)
                for node in self.start_nodes
            ]
            concurrent.futures.wait(futures)
        
        return self.results
    
    def run(self, input_data: Any = None) -> Dict[str, Any]:
        """Run the pipeline with the specified execution mode."""
        self.results = {}
        self.execution_times = {}
        self.start_time = time.time()
        self.status = "running"
        
        self.logger.info(f"Starting pipeline '{self.name}' in {self.execution_mode.value} mode")
        
        try:
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                results = self._execute_sequential(input_data)
            else:
                results = self._execute_parallel(input_data)
                
            self.status = "completed"
            self.logger.info(f"Pipeline '{self.name}' completed successfully")
            
        except Exception as e:
            self.status = "failed"
            self.logger.error(f"Pipeline '{self.name}' failed: {str(e)}")
            raise PipelineError(f"Pipeline execution failed: {str(e)}")
        finally:
            self.end_time = time.time()
            
        return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about the last pipeline execution."""
        if not self.execution_times:
            return {"error": "Pipeline has not been executed yet"}
        
        total_time = self.end_time - self.start_time
        node_stats = {}
        
        for node_id, node in self.nodes.items():
            node_stats[node_id] = {
                "name": node.name,
                "status": node.metadata.status.value,
                "execution_time": self.execution_times.get(node_id, 0),
                "attempts": node.metadata.attempt,
                "errors": len(node.metadata.errors)
            }
        
        return {
            "pipeline_id": self.id,
            "pipeline_name": self.name,
            "status": self.status,
            "total_execution_time": total_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "node_stats": node_stats,
            "node_count": len(self.nodes),
            "results_count": len(self.results)
        }
    
    def save_state(self, directory: str = "./pipeline_states") -> str:
        """Save the current state of the pipeline."""
        node_states = {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        
        state = PipelineState(
            pipeline_id=self.id,
            pipeline_name=self.name,
            execution_mode=self.execution_mode.value,
            node_states=node_states,
            results=self.results,
            execution_times=self.execution_times,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status
        )
        
        filepath = state.save(directory)
        self.logger.info(f"Pipeline state saved to {filepath}")
        return filepath
    
    def visualize(self, filename: str = None, view: bool = False) -> str:
        """Generate a visualization of the pipeline as a graph."""
        if filename is None:
            filename = f"pipeline_{self.id}"
            
        dot = graphviz.Digraph(comment=f"Pipeline: {self.name}")
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Add nodes
        for node_id, node in self.nodes.items():
            # Use different colors based on node status
            color = "black"
            fillcolor = "white"
            
            if node.metadata.status == NodeStatus.COMPLETED:
                color = "green"
                fillcolor = "lightgreen"
            elif node.metadata.status == NodeStatus.FAILED:
                color = "red"
                fillcolor = "lightpink"
            elif node.metadata.status == NodeStatus.RUNNING:
                color = "blue"
                fillcolor = "lightblue"
            
            # Add execution time if available
            label = node.name
            if node.id in self.execution_times:
                label += f"\n({self.execution_times[node.id]:.2f}s)"
                
            dot.node(node_id, label, shape="box", style="filled", color=color, fillcolor=fillcolor)
        
        # Add edges for regular connections
        for node_id, node in self.nodes.items():
            for next_node in node.next_nodes:
                dot.edge(node_id, next_node.id)
        
        # Add edges for conditional connections (with dashed line)
        for node_id, node in self.nodes.items():
            for next_node, condition in node.conditional_next_nodes:
                dot.edge(node_id, next_node.id, label=condition.description, style="dashed")
        
        # Highlight start nodes
        for node in self.start_nodes:
            dot.node(f"start_{node.id}", "", shape="point")
            dot.edge(f"start_{node.id}", node.id, style="bold")
        
        # Render the graph
        output_file = dot.render(filename=filename, format="png", cleanup=True, view=view)
        return output_file


def create_node(func: Callable, name: str = "", description: str = "", node_id: str = None) -> Node:
    """Helper function to create a node."""
    if node_id is None:
        node_id = f"node_{uuid.uuid4().hex[:8]}"
    return Node(id=node_id, func=func, name=name, description=description)


def create_condition(func: Callable[[Any], bool], description: str = "") -> Condition:
    """Helper function to create a condition."""
    return Condition(func=func, description=description)


def create_validator(func: Callable[[Any], Tuple[bool, str]], description: str = "") -> Validator:
    """Helper function to create a validator."""
    return Validator(func=func, description=description)


def load_pipeline_from_state(state_file: str) -> Pipeline:
    """Load a pipeline from a saved state file."""
    state = PipelineState.load(state_file)
    
    # Create a new pipeline
    pipeline = Pipeline(
        name=state.pipeline_name,
        execution_mode=ExecutionMode(state.execution_mode)
    )
    
    pipeline.id = state.pipeline_id
    pipeline.results = state.results
    pipeline.execution_times = state.execution_times
    pipeline.start_time = state.start_time
    pipeline.end_time = state.end_time
    pipeline.status = state.status
    
    # Note: This only loads the structure and state, not the actual node functions
    # which would need to be reconnected
    
    return pipeline
