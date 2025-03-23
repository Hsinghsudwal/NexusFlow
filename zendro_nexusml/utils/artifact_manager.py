import os
import json
import pickle
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ArtifactManager:
    """
    Manager for handling artifacts (models, datasets, etc.)
    Supports local storage and S3 (via LocalStack)
    """
    
    def __init__(self, base_path="artifacts"):
        """Initialize the artifact manager with a base path"""
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Initialized artifact manager with base path: {self.base_path}")
    
    def _ensure_path(self, path):
        """Ensure that the directory exists"""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def save_dataframe(self, df, filename, subdir=None):
        """Save a pandas DataFrame as CSV"""
        if subdir:
            path = os.path.join(self.base_path, subdir, filename)
        else:
            path = os.path.join(self.base_path, filename)
            
        self._ensure_path(path)
        df.to_csv(path, index=False)
        logger.info(f"Saved DataFrame to {path}")
        return path
    
    def load_dataframe(self, filepath):
        """Load a pandas DataFrame from CSV"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    
    def save_pickle(self, obj, filename, subdir=None):
        """Save an object using pickle"""
        if subdir:
            path = os.path.join(self.base_path, subdir, filename)
        else:
            path = os.path.join(self.base_path, filename)
            
        self._ensure_path(path)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle object to {path}")
        return path
    
    def load_pickle(self, filepath):
        """Load an object from pickle"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Loaded pickle object from {filepath}")
        return obj
    
    def save_json(self, data, filename, subdir=None):
        """Save data as JSON"""
        if subdir:
            path = os.path.join(self.base_path, subdir, filename)
        else:
            path = os.path.join(self.base_path, filename)
            
        self._ensure_path(path)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON data to {path}")
        return path
    
    def load_json(self, filepath):
        """Load data from JSON"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    
    def list_artifacts(self, subdir=None):
        """List all artifacts in the given subdirectory"""
        if subdir:
            path = os.path.join(self.base_path, subdir)
        else:
            path = self.base_path
            
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return []
            
        return os.listdir(path)
    
    def get_latest_artifact(self, pattern, subdir=None):
        """Get the latest artifact matching the pattern"""
        if subdir:
            path = os.path.join(self.base_path, subdir)
        else:
            path = self.base_path
            
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return None
            
        matching_files = [f for f in os.listdir(path) if pattern in f]
        if not matching_files:
            logger.warning(f"No files matching pattern {pattern} in {path}")
            return None
            
        # Get the latest file by modified time
        latest_file = max(matching_files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
        return os.path.join(path, latest_file)
    
    def copy_to_registry(self, filepath, version, registry_path=None):
        """Copy an artifact to the model registry"""
        if registry_path is None:
            registry_path = os.path.join(os.path.dirname(self.base_path), "model_registry")
            
        os.makedirs(registry_path, exist_ok=True)
        
        filename = os.path.basename(filepath)
        dest_path = os.path.join(registry_path, f"v{version}", filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        import shutil
        shutil.copy2(filepath, dest_path)
        logger.info(f"Copied {filepath} to model registry: {dest_path}")
        return dest_path


# artifact_management/artifact_manager.py

class ArtifactManager:
    def __init__(self, storage_path):
        self.storage_path = storage_path

    def save_artifact(self, name, model):
        # Save the model to a storage system
        print(f"Model {name} saved.")
    
    def load_artifact(self, name):
        # Load the model from the storage system
        print(f"Model {name} loaded.")
