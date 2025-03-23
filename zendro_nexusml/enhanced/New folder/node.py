import time
import threading

class Node:
    def __init__(self, name, task_fn):
        self.name = name  # Node name
        self.task_fn = task_fn  # Function to execute
        self.dependencies = []  # List of dependent nodes
        self.executed = False  # Flag to mark if the node has been executed

    def add_dependency(self, node):
        self.dependencies.append(node)

    def execute(self):
        """Executes the task for this node."""
        if not self.executed:
            print(f"Executing {self.name}...")
            self.task_fn()
            self.executed = True
        else:
            print(f"{self.name} has already been executed.")


import time
from nexusml.core.step import Step

class Node:
    def __init__(self, name, task_fn, dependencies=None):
        self.name = name  # Node name
        self.task_fn = task_fn  # Function to execute
        self.dependencies = dependencies or []  # List of dependent nodes
        self.executed = False  # Flag to mark if the node has been executed
        self.step = Step(name=name, dependencies=[dep.name for dep in self.dependencies], run=self.task_fn)

    def execute(self, executor, input_artifacts, config=None):
        """Executes the task for this node using the distributed executor."""
        if not self.executed:
            future = executor.execute_step(self.step, input_artifacts, config)
            future.add_done_callback(self._completion_callback)
        else:
            print(f"{self.name} has already been executed.")

    def _completion_callback(self, future):
        """Callback method when the node has completed execution."""
        try:
            future.result()  # Ensure any exceptions are propagated
            self.executed = True
            print(f"{self.name} completed successfully.")
        except Exception as e:
            print(f"Error executing {self.name}: {e}")
            self.executed = False
