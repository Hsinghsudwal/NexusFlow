import time
import logging
from typing import Callable, List, Dict, Any

logger = logging.getLogger(__name__)

class Step:
    def __init__(self, name: str, function: Callable, dependencies: List[str] = None):
        self.name = name
        self.function = function
        self.dependencies = dependencies or []
        self.start_time = None
        self.end_time = None
        self.inputs = {}
        self.outputs = {}
        self.metadata = {}

    def run(self, storage):
        """Execute the step with artifact storage."""
        self.start_time = time.time()
        
        # Load dependencies
        inputs = {dep: storage.load(dep) for dep in self.dependencies}
        self.inputs = inputs

        try:
            logger.info(f"Starting step: {self.name}")
            
            # Execute the function directly
            result = self.function(**inputs) if inputs else self.function()
            self.outputs = result if isinstance(result, dict) else {"result": result}
            
            # Save output
            storage.save(self.name, self.outputs)

            # Log metadata
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.metadata["duration"] = duration
            logger.info(f"Step {self.name} completed in {duration:.2f} seconds")
            
            return self.outputs

        except Exception as e:
            self.end_time = time.time()
            self.metadata["error"] = str(e)
            logger.error(f"Error in step {self.name}: {e}")
            raise
