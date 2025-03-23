import concurrent.futures
import uuid
from typing import List, Dict, Optional, Callable, Union
import time  # For simulating execution time of tasks
import logging
from datetime import datetime
import networkx as nx  # Importing networkx for DAG representation
import json
import os

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Experiment:
    def __init__(
        self,
        pipeline_id: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize an experiment for tracking pipeline execution.

        Args:
            pipeline_id (str, optional): The unique ID of the pipeline (if not provided, will generate a new one).
            pipeline_name (str, optional): The name of the pipeline.
            start_time (datetime, optional): The start time of the experiment (current time if not provided).
            description (str, optional): Optional description for the experiment.
        """
        self.pipeline_id = pipeline_id
        self.pipeline_name = pipeline_name
        self.start_time = start_time
        self.end_time = None
        self.status = "running"
        self.description = description
        self.metadata = {}

    def update_status(self, status: str):
        """Update the status of the experiment."""
        self.status = status

    def set_end_time(self, end_time: datetime):
        """Set the end time of the experiment."""
        self.end_time = end_time

    def save(self):
        """Save the experiment details to a JSON file (could also be a database)."""
        experiment_data = {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "description": self.description,
            "metadata": self.metadata,
        }
        # Save to a file (for simplicity here, you can change this to a database or other storage)
        data='data'
        experiment = 'experiment'
        
        experiment_path = os.path.join(data,experiment)
        os.makedirs(experiment_path, exist_ok=True)


        experiment_file = os.path.join(experiment_path, f"experiment_{experiment_data['pipeline_id']}.json")

        with open(experiment_file, "w") as f:
            json.dump(experiment_data, f, indent=4)
        
        print(f"Experiment data saved to {experiment_file}")

    def add_metadata(self, key: str, value: str):
        """Add metadata to the experiment."""
        self.metadata[key] = value


class Node:
    """Represents a node in the pipeline. Can also be used as a decorator."""

    def __init__(
        self,
        func: Optional[Callable] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize the node with a function and its attributes.

        Args:
            func (Callable, optional): The function to be wrapped by this node.
            inputs (List[str], optional): List of input artifacts required by the node.
            outputs (Union[str, List[str]], optional): List or single output artifact produced by the node.
            name (str, optional): Name of the node.
            tags (List[str], optional): Tags associated with the node.
        """
        if func is not None:
            self.func = func
            self.name = name or func.__name__
            self.inputs = inputs or []
            self.outputs = [outputs] if isinstance(outputs, str) else (outputs or [])
            self.tags = tags or []
            self.executed = False  # Track whether the node has been executed
        else:
            self.func = None
            self.name = name
            self.inputs = inputs or []
            self.outputs = outputs or []
            self.tags = tags or []

    def __call__(self, *args, **kwargs):
        """Allow the Node class to be used as a decorator."""
        return Node(
            func=self.func,
            inputs=self.inputs,
            outputs=self.outputs,
            name=self.name,
            tags=self.tags,
        )

    def execute(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Execute the node's task."""
        start_time = datetime.now()  # Initialize the start time when the node begins execution
        print(f"Executing node: {self.name}")
        time.sleep(1)  # Simulate execution time, replace with actual logic
        self.executed = True
        print(f"Node {self.name} execution complete.")

        duration = datetime.now() - start_time
        logger.info(f"Node {self.name} completed in {duration.total_seconds():.2f}s")


class Pipeline:
    def __init__(self, name: str, description: str, max_workers: Optional[int] = None):
        """
        Initialize the pipeline.

        Args:
            name (str): Name of the pipeline.
            description (str): Description of the pipeline.
            max_workers (int, optional): Maximum number of workers for parallel execution.
        """
        self.name = name
        self.description = description
        self.nodes: List[Node] = []  # List of nodes in the pipeline
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            if max_workers
            else None
        )
        #  self.steps = {}  # For the second pipeline (Step-based)
        self.dag = nx.DiGraph()  # For the directed acyclic graph representation
        self.id = str(uuid.uuid4())  # Unique pipeline ID for tracking

    def __call__(self, pipeline_class):
        """This method allows the Pipeline to be used as a decorator."""
        # Add any logic to populate the pipeline's nodes based on the decorated class
        pipeline_class.pipeline = self
        return pipeline_class

    def add_node(self, node: Node, dependencies: Optional[List[Node]] = None):
        """Add a node to the pipeline."""
        self.nodes.append(node)
        self.dag.add_node(node.name)

        # Add dependencies to the DAG
        if dependencies:
            for dep in dependencies:
                self.dag.add_edge(dep.name, node.name)

        # Verify that the DAG remains acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Pipeline contains a cycle in the node dependencies.")

    def run(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Run the pipeline and create an experiment to track the execution."""
        logger.info(f"Running pipeline '{self.name}'...")

        # Create the experiment object
        experiment = Experiment(
            pipeline_id=self.id,
            pipeline_name=self.name,
            start_time=datetime.now(),
            description=self.description,
        )

        # Use topological sorting to respect dependencies
        sorted_nodes = list(nx.topological_sort(self.dag))

        try:
            if self.executor:
                self._run_in_parallel(sorted_nodes, input_artifacts, config)
            else:
                self._run_sequentially(sorted_nodes, input_artifacts, config)

            # Mark experiment as completed
            experiment.update_status("completed")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            experiment.update_status("failed")

        # Set the end time and save the experiment
        experiment.set_end_time(datetime.now())
        experiment.save()

    def _run_sequentially(
        self,
        sorted_nodes: List[str],
        input_artifacts: Dict,
        config: Optional[Dict] = None,
    ):
        """Run nodes one after another (sequential execution)."""
        for node_name in sorted_nodes:
            node = next(n for n in self.nodes if n.name == node_name)
            logger.info(f"Executing node '{node.name}' sequentially...")
            self._execute_node_with_dependencies(node, input_artifacts, config)
            node.execute(input_artifacts, config)

    def _run_in_parallel(
        self,
        sorted_nodes: List[str],
        input_artifacts: Dict,
        config: Optional[Dict] = None,
    ):
        """Run nodes concurrently using ThreadPoolExecutor (parallel execution)."""
        futures = []

        for node_name in sorted_nodes:
            node = next(n for n in self.nodes if n.name == node_name)
            logger.info(f"Executing node '{node.name}' in parallel...")
            self._execute_node_with_dependencies(node, input_artifacts, config)

            # Submit node execution for parallel execution
            future = self.executor.submit(node.execute, input_artifacts, config)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will block until the task is completed

    def _execute_node_with_dependencies(
        self, node: Node, input_artifacts: Dict, config: Optional[Dict] = None
    ):
        """Ensure that all dependencies of a node are executed before running the node itself."""
        for dep_name in self.dag.predecessors(node.name):
            dep_node = next(n for n in self.nodes if n.name == dep_name)
            if not dep_node.executed:
                logger.info(f"Executing dependency: {dep_node.name}")
                dep_node.execute(input_artifacts, config)

    def shutdown(self):
        """Shutdown the executor after pipeline execution."""
        if self.executor:
            self.executor.shutdown()

