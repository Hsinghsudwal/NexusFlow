"""Node definition module."""
from typing import Dict, Any, Callable, Optional
import uuid
from datetime import datetime
import logging

from nexusml.core.artifact import Artifact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Node:
    def __init__(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None
        retries: int = 3
    ):
        """Initialize a pipeline step.

        Args:
            name: Name of the step
            func: Function to execute
            description: Optional description of the step
        """
        self.name = name
        self.func = func
        self.description = description
        self.id = str(uuid.uuid4())
        self.artifacts: Dict[str, Artifact] = {}
        self.retries = retries

    def run_node(
        self,
        input_artifacts: Dict[str, Any],
        config: Dict[str, Any] = None
    ) -> None:
        """Execute the step function.

        Args:
            input_artifacts: Dictionary of input artifacts
            config: Optional configuration parameters
        """
        start_time = datetime.now()
        logger.info(f"Starting step: {self.name}")
        logger.info(f"Input artifacts: {list(input_artifacts.keys())}")

        attempt = 0
        while attempt < self.retries:

            try:
                # Execute step function
                outputs = self.func(input_artifacts, config)

                # Store outputs as artifacts
                if outputs:
                    logger.info(f"Step {self.name} produced outputs: {list(outputs.keys())}")
                    for name, value in outputs.items():
                        self.artifacts[name] = Artifact(
                            name=name,
                            value=value,
                            step_id=self.id,
                            created_at=datetime.now()
                        )
                else:
                    logger.warning(f"Step {self.name} produced no outputs")

                logger.info(f"Step {self.name} completed.")
                break  # Exit retry loop if successful

            except Exception as e:
                attempt += 1
                logger.error(f"Step {self.name} failed on attempt {attempt}: {str(e)}")
                if attempt < self.retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Step {self.name} failed after {self.retries} attempts.")
                    raise RuntimeError(f"Step {self.name} failed after {self.retries} attempts.")


        duration = datetime.now() - start_time
        logger.info(f"Completed step: {self.name} in {duration.total_seconds():.2f}s")
