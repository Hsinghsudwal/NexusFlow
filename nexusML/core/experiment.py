import os
import sqlite3
from datetime import datetime
from typing import Optional
import json

class Experiment:
    def __init__(
        self,
        pipeline_id: str,
        pipeline_name: str,
        start_time: datetime,
        status: str = "running",
        error: Optional[str] = None,
        end_time: Optional[datetime] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            pipeline_id: ID of the pipeline
            pipeline_name: Name of the pipeline
            start_time: Start timestamp of the experiment
            status: Experiment status, default is "running"
            error: Optional error message if the experiment fails
            end_time: Optional end timestamp of the experiment
        """
        self.pipeline_id = pipeline_id
        self.pipeline_name = pipeline_name
        self.start_time = start_time
        self.status = status
        self.error = error
        self.end_time = end_time

        # Initialize SQLite database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for experiment tracking."""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Create experiments table if it does not exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                pipeline_id TEXT,
                pipeline_name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                error TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _get_db_path(self) -> str:
        """Return the path for the SQLite database."""
        return os.path.join(os.getcwd(), "nexusml.db")

    def save(self) -> None:
        """Save experiment details to the SQLite database."""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Insert experiment data into the database
        c.execute("""
            INSERT INTO experiments
            (pipeline_id, pipeline_name, start_time, end_time, status, error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self.pipeline_id,
            self.pipeline_name,
            self.start_time,
            self.end_time,
            self.status,
            self.error
        ))

        conn.commit()
        conn.close()

    @staticmethod
    def list_experiments() -> list:
        """Retrieve all experiments from the database."""
        db_path = Experiment._get_db_path(None)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("SELECT * FROM experiments")
        experiments = c.fetchall()

        conn.close()
        return experiments

    def save_to_json(self, filename: Optional[str] = None) -> None:
        """
        Save experiment details to a JSON file.

        Args:
            filename: Optional filename to save the experiment details to.
                      If not provided, defaults to "{self.pipeline_name}_experiment.json"
        """
        experiment_data = {
            'pipeline_id': self.pipeline_id,
            'pipeline_name': self.pipeline_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'error': self.error,
            'metrics': self.metrics if hasattr(self, 'metrics') else {},
            'hyperparameters': self.hyperparameters if hasattr(self, 'hyperparameters') else {}
        }

        if not filename:
            filename = f"{self.pipeline_name}_experiment.json"

        with open(filename, 'w') as f:
            json.dump(experiment_data, f, indent=4)

# Example usage:

# Initialize experiment
# experiment = Experiment(
#     pipeline_id="1234",
#     pipeline_name="sample_pipeline",
#     start_time=datetime.now()
# )

# Save experiment to SQLite
# experiment.save()

# List all experiments from SQLite
# experiments = Experiment.list_experiments()
# print(experiments)

# Save experiment to JSON
# experiment.save_to_json()
