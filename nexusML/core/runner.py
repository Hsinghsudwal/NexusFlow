from .pipeline import Pipeline
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class Runner:
    """Class responsible for running a pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        """
        Initialize the Runner with a pipeline.

        Args:
            pipeline (Pipeline): The pipeline instance to run.
        """
        self.pipeline = pipeline


    def start(self) -> None:
        """
        Start the execution of the pipeline.
        Logs the execution and invokes the pipeline's `run` method.
        """
        logger.info(f"Starting pipeline: {self.pipeline.name}")
        try:
            self.pipeline.run()
            logger.info(f"Pipeline '{self.pipeline.name}' completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline '{self.pipeline.name}' failed: {str(e)}")
            raise

# Assuming 'pipeline' is an instance of the Pipeline class
# runner = Runner(pipeline)
# runner.start()
