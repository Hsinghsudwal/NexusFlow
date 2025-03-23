"""
Handle parallel execution of independent pipeline steps
Manage task scheduling and dependencies
Provide status tracking for distributed tasks
First, let's create the necessary directory structure and base module.
"""


"""Distributed execution engine for pipeline steps."""
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Set
import logging
from datetime import datetime
from time import sleep
from nexusml.core.step import Step

logger = logging.getLogger(__name__)

class DistributedExecutor:
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize distributed executor.

        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[str, Future] = {}
        self.completed_steps: Set[str] = set()
        # self.step_status: Dict[str, str] = {}  # Track the status of each step

    def can_execute_step(self, step_name: str, dependencies: List[str]) -> bool:
        """Check if a step is ready for execution.

        Args:
            step_name: Name of the step to check
            dependencies: List of dependency step names

        Returns:
            True if step can be executed
        """
        return all(dep in self.completed_steps for dep in dependencies)

    def execute_step(self, step: Step, input_artifacts: Dict, config: Optional[Dict] = None) -> Future:
        """Submit a step for execution.

        Args:
            step: Step to execute
            input_artifacts: Input artifacts for the step
            config: Optional configuration

        Returns:
            Future object representing the pending execution
        """
        # Submit step execution
        future = self.executor.submit(step.run, input_artifacts, config)
        self.futures[step.name] = future

        # Add completion callback
        def _handle_completion(future: Future, step_name: str = step.name):
            try:
                # Get result to propagate any exceptions
                future.result()
                logger.info(f"Step {step_name} completed successfully")
                self.completed_steps.add(step_name)
            except Exception as e:
                logger.error(f"Step {step_name} failed: {str(e)}")
                raise

        future.add_done_callback(_handle_completion)
        return future


    # def _execute_with_tracking(self, step: Step, input_artifacts: Dict, config: Optional[Dict]):
    #     """Helper method to run the step and log execution status."""
    #     try:
    #         self.step_status[step.name] = 'RUNNING'
    #         logger.info(f"Executing step: {step.name}")
    #         step.run(input_artifacts, config)  # Execute the step's `run` method
    #         logger.info(f"Step {step.name} ran successfully.")
    #     except Exception as e:
    #         self.step_status[step.name] = 'FAILED'
    #         logger.error(f"Error executing step {step.name}: {str(e)}")
    #         raise  # Re-raise exception to propagate failure
    

    def wait_for_completion(self, timeout: Optional[float] = None) -> None:
        """Wait for all submitted steps to complete.

        Args:
            timeout: Maximum time to wait in seconds
        """
        try:
            # Wait for all futures to complete
            for name, future in self.futures.items():
                future.result(timeout=timeout)
                logger.info(f"Verified completion of step: {name}")

        except Exception as e:
            logger.error(f"Error waiting for steps to complete: {str(e)}")
            raise

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the executor and handle pending tasks.

        Args:
            wait: Whether to wait for all tasks to complete before shutting down
        """
        logger.info("Shutting down executor...")
        self.executor.shutdown(wait=wait)
        logger.info("Executor shutdown complete")

    def get_step_status(self, step_name: str) -> Optional[str]:
        """Get the current status of a specific step.

        Args:
            step_name: Name of the step

        Returns:
            The status of the step ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')
        """
        return self.step_status.get(step_name, 'UNKNOWN')

    def list_pending_steps(self) -> List[str]:
        """List all steps that are currently pending execution.

        Returns:
            List of names of pending steps
        """
        return [step for step, status in self.step_status.items() if status == 'PENDING']

    def list_running_steps(self) -> List[str]:
        """List all steps that are currently running.

        Returns:
            List of names of running steps
        """
        return [step for step, status in self.step_status.items() if status == 'RUNNING']

    def list_completed_steps(self) -> List[str]:
        """List all steps that have completed execution.

        Returns:
            List of names of completed steps
        """
        return [step for step, status in self.step_status.items() if status == 'COMPLETED']

    def list_failed_steps(self) -> List[str]:
        """List all steps that have failed execution.

        Returns:
            List of names of failed steps
        """
        return [step for step, status in self.step_status.items() if status == 'FAILED']



# from distributed_executor import DistributedExecutor
# from nexusml.core.step import Step  # Assuming a Step class exists
# from datetime import datetime

# Initialize executor with 4 concurrent workers
# executor = DistributedExecutor(max_workers=4)

# Example step
# step = Step(name="step_1", dependencies=[], run=lambda x, y: print("Running step_1"))

# Example input artifacts and config (can be empty in this case)
# input_artifacts = {}
# config = {}

# Execute the step
# future = executor.execute_step(step, input_artifacts, config)

# Wait for all tasks to complete
# executor.wait_for_completion()

# Check status of a step
# status = executor.get_step_status("step_1")
# print(f"Step status: {status}")

# Gracefully shut down the executor
# executor.shutdown(wait=True)