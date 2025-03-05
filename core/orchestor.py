class Orchestrator:
    """Manages the scheduling and execution of pipelines."""

    def __init__(self):
        """Initialize the orchestrator with an empty list of scheduled pipelines."""
        self.scheduled_pipelines = []

    def schedule(self, pipeline) -> None:
        """
        Schedule a pipeline for execution.
        Args:
            pipeline: The pipeline object to schedule.
        """
        self.scheduled_pipelines.append(pipeline)
        print(f"Pipeline {pipeline.name} scheduled.")

    def execute_scheduled_pipelines(self):
        """
        Execute all scheduled pipelines in sequence.
        Each pipeline is run using a separate runner.
        """
        if not self.scheduled_pipelines:
            print("No pipelines are scheduled for execution.")
            return

        for pipeline in self.scheduled_pipelines:
            runner = Runner(pipeline)
            runner.start()

        print(f"Executed {len(self._scheduled_pipelines)} scheduled pipeline(s).")

    def list_scheduled_pipelines(self) -> list:
        """
        List all scheduled pipelines.

        Returns:
            list: A list of pipeline names that are scheduled for execution.
        """
        return [pipeline.name for pipeline in self._scheduled_pipelines]

# Example usage:

# Create an orchestrator instance
# orchestrator = Orchestrator()

# Assume 'pipeline1' and 'pipeline2' are pipeline objects with a 'name' attribute
# orchestrator.schedule(pipeline1)
# orchestrator.schedule(pipeline2)

# Execute scheduled pipelines
# orchestrator.execute_scheduled_pipelines()

# List all scheduled pipelines
# print(orchestrator.list_scheduled_pipelines())
