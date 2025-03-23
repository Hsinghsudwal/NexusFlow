# mlops_framework/pipeline.py

import inspect
import os
import uuid
import datetime
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from .metadata_store import MetadataStore
from .artifact_store import ArtifactStore
from .config import ConfigManager

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Pipeline class to orchestrate ML workflows.
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initialize a pipeline.
        
        Args:
            name: Name of the pipeline
            description: Optional description
        """
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())
        self.steps = []
        self.config = ConfigManager()
        self.metadata_store = MetadataStore()
        self.artifact_store = ArtifactStore()
        self.start_time = None
        self.end_time = None
        
    def add_step(self, step_fn: Callable):
        """
        Add a step to the pipeline.
        
        Args:
            step_fn: Step function to add
        """
        self.steps.append(step_fn)
        return self
        
    def run(self, **kwargs):
        """
        Run the pipeline with the given parameters.
        
        Args:
            **kwargs: Parameters to pass to the steps
        
        Returns:
            Dict of outputs from each step
        """
        self.start_time = datetime.datetime.now()
        logger.info(f"Starting pipeline '{self.name}'")
        
        # Register pipeline run in metadata store
        run_id = self.metadata_store.register_pipeline_run(
            pipeline_id=self.id, 
            pipeline_name=self.name,
            start_time=self.start_time
        )
        
        results = {}
        step_inputs = kwargs.copy()
        
        try:
            for step_fn in self.steps:
                step_name = step_fn.__name__
                step_id = getattr(step_fn, "step_id", str(uuid.uuid4()))
                
                logger.info(f"Executing step '{step_name}'")
                
                # Register step execution in metadata store
                step_run_id = self.metadata_store.register_step_run(
                    run_id=run_id,
                    step_id=step_id,
                    step_name=step_name,
                    start_time=datetime.datetime.now()
                )
                
                # Get required inputs for this step
                sig = inspect.signature(step_fn)
                step_kwargs = {}
                
                for param_name, param in sig.parameters.items():
                    if param_name in step_inputs:
                        step_kwargs[param_name] = step_inputs[param_name]
                
                # Execute the step
                try:
                    output = step_fn(**step_kwargs)
                    
                    # Record successful execution
                    self.metadata_store.update_step_run(
                        step_run_id=step_run_id,
                        status="success",
                        end_time=datetime.datetime.now()
                    )
                    
                    # Store the output in results and make available for next steps
                    results[step_name] = output
                    step_inputs[step_name] = output
                    
                except Exception as e:
                    # Record failed execution
                    self.metadata_store.update_step_run(
                        step_run_id=step_run_id,
                        status="failed",
                        end_time=datetime.datetime.now(),
                        error=str(e)
                    )
                    raise
                
            self.end_time = datetime.datetime.now()
            
            # Update pipeline run in metadata store
            self.metadata_store.update_pipeline_run(
                run_id=run_id,
                status="success",
                end_time=self.end_time
            )
            
            logger.info(f"Pipeline '{self.name}' completed successfully")
            return results
            
        except Exception as e:
            self.end_time = datetime.datetime.now()
            
            # Update pipeline run in metadata store
            self.metadata_store.update_pipeline_run(
                run_id=run_id,
                status="failed",
                end_time=self.end_time,
                error=str(e)
            )
            
            logger.error(f"Pipeline '{self.name}' failed with error: {str(e)}")
            raise

def pipeline(name: str, description: Optional[str] = None):
    """
    Decorator to create a pipeline.
    
    Args:
        name: Name of the pipeline
        description: Optional description
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pipe = Pipeline(name=name, description=description)
            
            # Call the decorated function to set up the pipeline
            func(pipe, *args, **kwargs)
            
            # Run the pipeline
            return pipe.run(**kwargs)
        
        return wrapper
    
    return decorator
