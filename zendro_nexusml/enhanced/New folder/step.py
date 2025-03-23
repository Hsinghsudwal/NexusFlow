# step.py
import time
import logging
from datetime import datetime
from artifact_storage import ArtifactStorage
import uuid

logger = logging.getLogger(__name__)

class Step:
    def __init__(self, name: str, fn: callable, retries: int = 3):
        self.name = name
        self.fn = fn
        self.id = str(uuid.uuid4())
        self.retries = retries

    def run(self, input_artifacts):
        """Run the step function with retry logic."""
        logger.info(f"Starting step: {self.name}")
        start_time = datetime.now()

        # Instantiate ArtifactStorage (use values from constants.py)
        artifact_storage = ArtifactStorage()

        attempt = 0
        while attempt < self.retries:
            try:
                outputs = self.fn(input_artifacts)

                # Save the outputs as artifacts
                if outputs:
                    for artifact_name, artifact_value in outputs.items():
                        artifact_location = artifact_storage.save(artifact_name, artifact_value)
                        logger.info(f"Artifact saved at: {artifact_location}")

                logger.info(f"Step {self.name} completed.")
                return outputs
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
        logger.info(f"Step {self.name} completed in {duration.total_seconds():.2f}s")
