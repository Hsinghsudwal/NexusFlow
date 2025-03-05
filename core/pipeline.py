"""Pipeline orchestration module."""
import networkx as nx
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid
import logging
import time

from nexusml.core.step import Step
from nexusml.core.artifact import Artifact
from nexusml.tracking.experiment import Experiment
from nexusml.distributed.executor import DistributedExecutor

from nexusml.storage.base import StorageProvider, LocalStorageProvider
from nexusml.storage.cloud import S3StorageProvider

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        distributed: bool = False,
        max_workers: Optional[int] = None
        storage_provider: Optional[StorageProvider] = None
    ):
        """Initialize a new pipeline.

        Args:
            name: Name of the pipeline
            description: Optional description of the pipeline
            distributed: Enable distributed execution
            max_workers: Maximum number of concurrent workers for distributed execution
        """
        self.name = name
        self.description = description
        self.steps: Dict[str, Step] = {}
        self.dag = nx.DiGraph()
        self.id = str(uuid.uuid4())
        self.artifacts: Dict[str, Artifact] = {}
        self.distributed = distributed
        self.executor = DistributedExecutor(max_workers) if distributed else None
        self.storage_provider = storage_provider


    def add_step(self, step: Step, dependencies: Optional[List[Step]] = None) -> None:
        """Add a step to the pipeline with optional dependencies.

        Args:
            step: Step to add
            dependencies: List of steps that must complete before this step
        """
        self.steps[step.name] = step
        self.dag.add_node(step.name)

        if dependencies:
            for dep in dependencies:
                self.dag.add_edge(dep.name, step.name)

        # Verify acyclic
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Pipeline steps contain a cycle")

    def _get_step_dependencies(self, step_name: str) -> List[str]:
        """Get names of immediate dependencies for a step."""
        return list(self.dag.predecessors(step_name))

    def _step_ready_to_execute(self, step_name: str, completed_steps: Set[str]) -> bool:
        """Check if a step is ready to execute based on completed dependencies."""
        deps = self._get_step_dependencies(step_name)
        return all(dep in completed_steps for dep in deps)

    def run(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Execute the pipeline steps in order.

        Args:
            config: Optional configuration parameters
        """
        # Create experiment run
        experiment = Experiment(
            pipeline_id=self.id,
            pipeline_name=self.name,
            start_time=datetime.now()
        )

        try:
            if self.distributed and self.executor:
                logger.info("Running pipeline in distributed mode")

                # Get execution order and initialize pending steps
                execution_order = list(nx.topological_sort(self.dag))
                pending_steps = set(execution_order)
                scheduled_steps = set()

                while pending_steps:
                    # Find ready steps
                    for step_name in list(pending_steps):  # Create a copy to avoid modification during iteration
                        if self._step_ready_to_execute(step_name, self.executor.completed_steps):
                            step = self.steps[step_name]

                            # Collect input artifacts
                            input_artifacts = {}
                            for dep in self._get_step_dependencies(step_name):
                                for key, artifact in self.steps[dep].artifacts.items():
                                    input_artifacts[key] = artifact.value

                            # Schedule step execution
                            self.executor.execute_step(step, input_artifacts, config)
                            pending_steps.remove(step_name)
                            scheduled_steps.add(step_name)
                            logger.info(f"Scheduled step: {step_name}")

                    # Wait for some steps to complete
                    time.sleep(0.1)  # Prevent CPU thrashing

                    # Update completed steps
                    newly_completed = scheduled_steps - self.executor.completed_steps
                    if newly_completed:
                        logger.info(f"Waiting for steps to complete: {newly_completed}")

                # Wait for all steps to complete
                self.executor.wait_for_completion()

                # Collect artifacts from all steps
                for step_name in execution_order:
                    self.artifacts.update(self.steps[step_name].artifacts)

            else:
                logger.info("Running pipeline in sequential mode")
                # Execute steps sequentially
                for step_name in nx.topological_sort(self.dag):
                    step = self.steps[step_name]

                    # Get input artifacts from dependencies
                    input_artifacts = {}
                    for dep in self.dag.predecessors(step_name):
                        for key, artifact in self.steps[dep].artifacts.items():
                            input_artifacts[key] = artifact.value

                    # Run step
                    step.run(input_artifacts, config)

                    # Store artifacts
                    self.artifacts.update(step.artifacts)

            experiment.status = "completed"

        except Exception as e:
            experiment.status = "failed"
            experiment.error = str(e)
            raise

        finally:
            experiment.end_time = datetime.now()
            experiment.save()
