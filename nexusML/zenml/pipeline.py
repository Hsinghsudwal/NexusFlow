import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, name: str, steps: List['Step'], parallel: bool = False):
        """
        Initializes the pipeline.
        :param steps: List of Step instances.
        :param parallel: Whether to execute steps in parallel.
        """
        self.steps = steps
        self.parallel = parallel
        self.name = name
        self.artifact_store = ArtifactStore()
        self.execution_graph = {}
        self.results = {}

    def add_step(self, step: 'Step', dependencies: List[str] = None) -> None:
        """Add a step to the pipeline with optional dependencies."""
        self.steps.append(step)
        if dependencies:
            self.execution_graph[step.name] = dependencies

    def run(self, initial_inputs: Dict = None) -> Dict:
        """Run the pipeline."""
        try:
            start_time = time.time()  # Initialize start time for duration tracking
            self.results = initial_inputs or {}

            if self.parallel:
                # Parallel execution of steps
                storage = ArtifactStorage()
                with ThreadPoolExecutor() as executor:
                    # Run steps in parallel, each step will be given the storage to handle artifacts
                    results = list(executor.map(lambda step: step.run(storage), self.steps))
                logger.info(f"Pipeline {self.name} completed in {time.time() - start_time:.2f} seconds")
                return results
            else:
                # Sequential execution of steps with dependency handling
                logger.info(f"Starting pipeline: {self.name}")
                pending_steps = self.steps.copy()
                completed_steps = set()

                while pending_steps:
                    next_steps = []
                    
                    for step in pending_steps:
                        # Check if dependencies are met
                        if step.name in self.execution_graph:
                            dependencies = self.execution_graph[step.name]
                            if not all(dep in completed_steps for dep in dependencies):
                                next_steps.append(step)
                                continue

                        # Execute the step sequentially
                        logger.info(f"Executing step: {step.name}")
                        step_inputs = {key: value for key, value in self.results.items()}
                        
                        # Execute the step with artifact storage
                        storage = ArtifactStorage()
                        step_results = step.run(storage)

                        # Store the results and artifacts
                        for key, value in step_results.items():
                            artifact_id = self.artifact_store.save_artifact(
                                value,
                                f"{step.name}_{key}",
                                {"step_id": step.name}  # Assuming step_id is the name for simplicity
                            )
                            self.results[key] = value

                        # Mark the step as completed
                        completed_steps.add(step.name)
                    
                    pending_steps = next_steps
                    
                    # Check for circular dependencies
                    if len(pending_steps) == len(next_steps) and next_steps:
                        raise RuntimeError(f"Circular dependency detected among steps: {[s.name for s in next_steps]}")

                logger.info(f"Pipeline {self.name} completed in {time.time() - start_time:.2f} seconds")
                return self.results

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
