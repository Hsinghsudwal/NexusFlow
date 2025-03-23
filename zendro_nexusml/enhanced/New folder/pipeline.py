class Pipeline:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.nodes = []  # List of all nodes in the pipeline
        self.execution_mode = 'sequential'  # Default mode

    def add_node(self, node):
        self.nodes.append(node)

    def set_execution_mode(self, mode):
        """Sets the execution mode of the pipeline: 'sequential' or 'parallel'."""
        if mode in ['sequential', 'parallel']:
            self.execution_mode = mode
        else:
            print("Invalid mode. Use 'sequential' or 'parallel'.")

    def run(self):
        """Runs the pipeline based on the selected mode."""
        if self.execution_mode == 'sequential':
            self._run_sequentially()
        elif self.execution_mode == 'parallel':
            self._run_in_parallel()

    def _run_sequentially(self):
        """Runs nodes one after another."""
        print(f"Running pipeline '{self.name}' in sequential mode...")
        for node in self.nodes:
            # Ensure all dependencies are executed before running the current node
            for dependency in node.dependencies:
                if not dependency.executed:
                    dependency.execute()
            node.execute()

    def _run_in_parallel(self):
        """Runs nodes concurrently."""
        print(f"Running pipeline '{self.name}' in parallel mode...")
        threads = []
        for node in self.nodes:
            # Ensure all dependencies are executed before running the current node
            for dependency in node.dependencies:
                if not dependency.executed:
                    dependency.execute()

            # Run each node in a separate thread for parallel execution
            thread = threading.Thread(target=node.execute)
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()



from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional
from distributed_executor import DistributedExecutor  # Assuming it's in a separate file

class Pipeline:
    def __init__(self, name, description, max_workers: Optional[int] = None):
        self.name = name
        self.description = description
        self.nodes = []  # List of all nodes in the pipeline
        self.executor = DistributedExecutor(max_workers=max_workers)

    def add_node(self, node):
        """Add a node to the pipeline."""
        self.nodes.append(node)

    def run(self, input_artifacts: Dict, config: Optional[Dict] = None):
        """Runs the pipeline, executing nodes based on dependencies."""
        print(f"Running pipeline '{self.name}'...")

        # Start execution of each node
        for node in self.nodes:
            # Check if all dependencies are completed
            if all(dep.executed for dep in node.dependencies):
                node.execute(self.executor, input_artifacts, config)

        # Wait for all tasks to complete
        self.executor.wait_for_completion()

    def shutdown(self):
        """Shutdown the executor after pipeline execution."""
        self.executor.shutdown()

