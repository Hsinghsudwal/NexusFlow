import logging
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional, Callable

class UnifiedPipeline:
    """A unified pipeline that combines task and pipeline functionality."""
    
    def __init__(self, name: str, config: Dict = None, description: str = ""):
        self.name = name
        self.description = description
        self.config = Config(config) if config is not None else {}
        self.tasks = []
        self.run_history = {}
        self.current_run_id = None
        self.artifact_store = ArtifactStore(self.config)
        logging.info(f"Pipeline '{name}' initialized")
    
    def add_task(self, task_function: Callable, name: str = None, description: str = "") -> None:
        """Add a task function to the pipeline.
        
        Args:
            task_function: The function implementing the task logic
            name: Name of the task (defaults to function name if not provided)
            description: Description of the task
        """
        task = {
            "function": task_function,
            "name": name or task_function.__name__,
            "description": description
        }
        self.tasks.append(task)
        logging.info(f"Task '{task['name']}' added to pipeline '{self.name}'")
    
    def _execute_task(self, task: Dict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task and record metrics."""
        task_name = task["name"]
        logging.info(f"Starting task: {task_name}")
        start_time = pd.Timestamp.now()
        
        try:
            # Execute the task function
            updated_context = task["function"](self, context)
            
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Task '{task_name}' completed in {duration:.2f} seconds")
            
            # Record task execution in the run history
            task_result = {
                "name": task_name,
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            }
            
            if self.current_run_id and self.current_run_id in self.run_history:
                self.run_history[self.current_run_id]["tasks"].append(task_result)
            
            return updated_context
            
        except Exception as e:
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            logging.error(f"Task '{task_name}' failed after {duration:.2f} seconds: {str(e)}")
            
            # Record failure
            task_result = {
                "name": task_name,
                "status": "failed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": str(e)
            }
            
            if self.current_run_id and self.current_run_id in self.run_history:
                self.run_history[self.current_run_id]["tasks"].append(task_result)
            
            raise
    
    def run(self, context: Dict[str, Any] = None) -> str:
        """Run all tasks in the pipeline and return the run ID."""
        self.current_run_id = f"run_{uuid.uuid4().hex[:8]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = pd.Timestamp.now()
        
        # Initialize run in history
        self.run_history[self.current_run_id] = {
            "pipeline_name": self.name,
            "start_time": start_time.isoformat(),
            "status": "running",
            "tasks": []
        }
        
        logging.info(f"Starting pipeline '{self.name}' with run ID: {self.current_run_id}")
        
        # Initialize context
        current_context = context or {}
        
        try:
            # Execute each task in sequence
            for task in self.tasks:
                current_context = self._execute_task(task, current_context)
                
            # Record successful completion
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            self.run_history[self.current_run_id].update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            })
            
            logging.info(f"Pipeline '{self.name}' completed in {duration:.2f} seconds")
            
            # Save run details as artifact
            self.artifact_store.save_artifact(
                self.run_history[self.current_run_id],
                subdir="runs",
                name=f"{self.current_run_id}_details.json",
                run_id=self.current_run_id
            )
            
            return self.current_run_id
            
        except Exception as e:
            # Record failure
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            self.run_history[self.current_run_id].update({
                "status": "failed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": str(e)
            })
            
            logging.error(f"Pipeline '{self.name}' failed after {duration:.2f} seconds: {str(e)}")
            
            # Save run details even for failed runs
            self.artifact_store.save_artifact(
                self.run_history[self.current_run_id],
                subdir="runs",
                name=f"{self.current_run_id}_details.json",
                run_id=self.current_run_id
            )
            
            raise
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get details about a specific pipeline run."""
        if run_id in self.run_history:
            return self.run_history[run_id]
        
        # Try to load from artifact store
        run_details = self.artifact_store.load_artifact(
            subdir="runs",
            name=f"{run_id}_details.json",
            run_id=run_id
        )
        
        if run_details:
            # Cache in memory
            self.run_history[run_id] = run_details
            return run_details
            
        logging.warning(f"No details found for run ID: {run_id}")
        return None


# Example usage
def create_pipeline(name: str, config: Dict = None) -> UnifiedPipeline:
    """Factory function to create a new pipeline."""
    return UnifiedPipeline(name=name, config=config)


# Decorator to create task functions
def task(name: str = None, description: str = ""):
    """Decorator to mark functions as pipeline tasks."""
    def decorator(func):
        # We'll keep the original function but add metadata
        func._task_metadata = {
            "name": name or func.__name__,
            "description": description
        }
        return func
    return decorator


# Example of how to use this unified approach
"""
# Example usage with decorators
@task(name="data_loading", description="Load data from source")
def load_data(pipeline, context):
    # Implement task logic here
    context['data'] = [1, 2, 3]  # Just an example
    return context

# Creating and running a pipeline
pipeline = create_pipeline("my_pipeline")
pipeline.add_task(load_data)

# Alternative way to add task without decorator
def process_data(pipeline, context):
    context['processed_data'] = [x * 2 for x in context['data']]
    return context

pipeline.add_task(process_data, name="data_processing", description="Process the loaded data")

# Run pipeline
result_id = pipeline.run()
"""
