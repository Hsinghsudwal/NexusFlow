import uuid
import json
from datetime import datetime
from typing import Optional, Dict


class Experiment:
    def __init__(self, pipeline_id: Optional[str] = None, pipeline_name: Optional[str] = None, 
                 start_time: Optional[datetime] = None, description: Optional[str] = None):
        """
        Initialize an experiment for tracking pipeline execution.

        Args:
            pipeline_id (str, optional): The unique ID of the pipeline (if not provided, will generate a new one).
            pipeline_name (str, optional): The name of the pipeline.
            start_time (datetime, optional): The start time of the experiment (current time if not provided).
            description (str, optional): Optional description for the experiment.
        """
        self.pipeline_id = pipeline_id or str(uuid.uuid4())  # Generate a unique ID if not provided
        self.pipeline_name = pipeline_name
        self.start_time = start_time or datetime.now()  # Use current time if not provided
        self.end_time = None
        self.status = "running"  # Initially the experiment is running
        self.description = description
        self.metadata = {}
        self.error = None
        self.metrics = {}
        self.hyperparameters = {}

    def update_status(self, status: str):
        """Update the status of the experiment."""
        self.status = status

    def set_end_time(self, end_time: datetime):
        """Set the end time of the experiment."""
        self.end_time = end_time

    def add_metadata(self, key: str, value: str):
        """Add metadata to the experiment."""
        self.metadata[key] = value

    def add_metric(self, metric_name: str, value: float):
        """Add a metric to the experiment."""
        self.metrics[metric_name] = value

    def add_hyperparameter(self, hyperparameter_name: str, value: str):
        """Add a hyperparameter to the experiment."""
        self.hyperparameters[hyperparameter_name] = value

    def set_error(self, error_message: str):
        """Record an error in the experiment."""
        self.error = error_message
        self.status = "failed"

    def save(self):
        """Save the experiment details to a JSON file."""
        experiment_data = {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "description": self.description,
            "error": self.error,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata
        }
        
        # Save the experiment to a JSON file
        with open(f"experiment_{self.pipeline_id}.json", "w") as f:
            json.dump(experiment_data, f, indent=4)
        
        print(f"Experiment saved to experiment_{self.pipeline_id}.json")




from datetime import datetime

# Create an experiment for the pipeline run
experiment = Experiment(
    pipeline_id="unique-pipeline-id",
    pipeline_name="Sample Pipeline",
    start_time=datetime.now(),
    description="This is a description of the experiment"
)

# Add hyperparameters, metrics, and metadata as the pipeline runs
experiment.add_hyperparameter("learning_rate", "0.01")
experiment.add_hyperparameter("n_estimators", "100")

# Simulate pipeline execution with some metrics
experiment.add_metric("accuracy", 0.95)

# Update experiment status when pipeline finishes
experiment.update_status("completed")
experiment.set_end_time(datetime.now())

# Save the experiment details to a JSON file
experiment.save()
