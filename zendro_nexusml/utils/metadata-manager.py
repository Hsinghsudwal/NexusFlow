import os
import json
import logging
import datetime
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class MetadataManager:
    """
    Manager for tracking metadata about pipeline runs
    """
    
    def __init__(self, metadata_path="metadata.json"):
        """Initialize the metadata manager with a path to store metadata"""
        self.metadata_path = metadata_path
        self.metadata = {
            "run_id": None,
            "start_time": None,
            "end_time": None,
            "status": None,
            "error": None,
            "mode": None,
            "steps": [],
        }
        logger.info(f"Initialized metadata manager with path: {metadata_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    def start_run(self, run_id=None, mode="full"):
        """Start a new pipeline run"""
        if run_id is None:
            run_id = str(uuid.uuid4())
            
        self.metadata["run_id"] = run_id
        self.metadata["start_time"] = datetime.datetime.now().isoformat()
        self.metadata["mode"] = mode
        self.metadata["status"] = "running"
        self.metadata["steps"] = []
        
        logger.info(f"Started pipeline run with ID: {run_id}")
        self._save_metadata()
        
        return run_id
    
    def end_run(self, status="completed", error=None):
        """End the pipeline run with status and optional error"""
        self.metadata["end_time"] = datetime.datetime.now().isoformat()
        self.metadata["status"] = status
        if error:
            self.metadata["error"] = error
            
        logger.info(f"Ended pipeline run with status: {status}")
        self._save_metadata()
    
    def log_step(self, step_name, artifacts=None, metrics=None, parameters=None):
        """Log metadata for a pipeline step"""
        step_data = {
            "name": step_name,
            "start_time": datetime.datetime.now().isoformat(),
            "artifacts": artifacts or {},
            "metrics": metrics or {},
            "parameters": parameters or {},
        }
        
        self.metadata["steps"].append(step_data)
        logger.info(f"Logged metadata for step: {step_name}")
        self._save_metadata()
        
        return step_data
    
    def update_step_metrics(self, step_name, metrics):
        """Update metrics for a specific step"""
        for step in self.metadata["steps"]:
            if step["name"] == step_name:
                step["metrics"].update(metrics)
                logger.info(f"Updated metrics for step: {step_name}")
                self._save_metadata()
                return True
                
        logger.warning(f"Step not found: {step_name}")
        return False
    
    def get_run_metadata(self):
        """Get the current run metadata"""
        return self.metadata
    
    def get_step_metadata(self, step_name):
        """Get metadata for a specific step"""
        for step in self.metadata["steps"]:
            if step["name"] == step_name:
                return step
                
        logger.warning(f"Step not found: {step_name}")
        return None
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
    def load_metadata(self, filepath=None):
        """Load metadata from file"""
        if filepath is None:
            filepath = self.metadata_path
            
        if not os.path.exists(filepath):
            logger.warning(f"Metadata file not found: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            self.metadata = json.load(f)
            
        logger.info(f"Loaded metadata from {filepath}")
        return self.metadata
    
    def list_runs(self, metadata_dir=None):
        """List all runs in the metadata directory"""
        if metadata_dir is None:
            metadata_dir = os.path.dirname(self.metadata_path)
            
        if not os.path.exists(metadata_dir):
            logger.warning(f"Metadata directory not found: {metadata_dir}")
            return []
            
        runs = []
        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(metadata_dir, filename)
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    runs.append({
                        "run_id": metadata.get("run_id"),
                        "start_time": metadata.get("start_time"),
                        "end_time": metadata.get("end_time"),
                        "status": metadata.get("status"),
                        "mode": metadata.get("mode")
                    })
                    
        return runs
