# mlops_framework/artifact_store.py

import os
import uuid
import json
import pickle
import logging
import shutil
from typing import Any, Dict, List, Optional, Union
import datetime

logger = logging.getLogger(__name__)

class ArtifactStore:
    """
    Singleton class to manage ML artifacts storage.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ArtifactStore, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the artifact store."""
        self.base_path = os.environ.get("ARTIFACT_STORE_PATH", "artifacts")
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Initialized artifact store at {self.base_path}")
        
        # Initialize storage backends
        self.storage_backend = self._get_storage_backend()
    
    def _get_storage_backend(self):
        """
        Get the configured storage backend based on environment variables.
        
        Returns:
            Storage backend instance
        """
        backend_type = os.environ.get("ARTIFACT_BACKEND", "local")
        
        if backend_type == "local":
            return LocalStorageBackend(self.base_path)
        elif backend_type == "s3":
            from .backends.s3_backend import S3StorageBackend
            bucket = os.environ.get("S3_BUCKET")
            prefix = os.environ.get("S3_PREFIX", "artifacts")
            return S3StorageBackend(bucket, prefix)
        elif backend_type == "gcs":
            from .backends.gcs_backend import GCSStorageBackend
            bucket = os.environ.get("GCS_BUCKET")
            prefix = os.environ.get("GCS_PREFIX", "artifacts")
            return GCSStorageBackend(bucket, prefix)
        elif backend_type == "azure":
            from .backends.azure_backend import AzureStorageBackend
            account = os.environ.get("AZURE_STORAGE_ACCOUNT")
            container = os.environ.get("AZURE_CONTAINER")
            prefix = os.environ.get("AZURE_PREFIX", "artifacts")
            return AzureStorageBackend(account, container, prefix)
        else:
            logger.warning(f"Unknown storage backend {backend_type}, using local storage")
            return LocalStorageBackend(self.base_path)
    
    def save_artifact(self, name: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Save an artifact.
        
        Args:
            name: Name of the artifact
            data: Artifact data
            metadata: Optional metadata
            
        Returns:
            Artifact ID
        """
        artifact_id = str(uuid.uuid4())
        
        # Generate a path for the artifact
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{name}_{timestamp}_{artifact_id}"
        
        # Save the artifact using the backend
        self.storage_backend.save(path, data)
        
        # Save metadata
        metadata = metadata or {}
        metadata["artifact_id"] = artifact_id
        metadata["name"] = name
        metadata["timestamp"] = timestamp
        metadata["path"] = path
        
        metadata_path = f"{path}_metadata.json"
        self.storage_backend.save_metadata(metadata_path, metadata)
        
        logger.info(f"Saved artifact {name} with ID {artifact_id}")
        return artifact_id
    
    def load_artifact(self, artifact_id: str) -> Any:
        """
        Load an artifact by ID.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            Artifact data
        """
        # Find the artifact by ID
        artifacts = self.list_artifacts()
        for artifact in artifacts:
            if artifact.get("artifact_id") == artifact_id:
                path = artifact.get("path")
                return self.storage_backend.load(path)
        
        raise ValueError(f"Artifact with ID {artifact_id} not found")
    
    def list_artifacts(self, name: Optional[str] = None) -> List[Dict]:
        """
        List available artifacts, optionally filtered by name.
        
        Args:
            name: Optional artifact name to filter by
            
        Returns:
            List of artifacts with their metadata
        """
        artifacts = self.storage_backend.list_artifacts()
        
        if name:
            artifacts = [a for a in artifacts if a.get("name") == name]
            
        return artifacts
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact by ID.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            True if deleted, False otherwise
        """
        # Find the artifact by ID
        artifacts = self.list_artifacts()
        for artifact in artifacts:
            if artifact.get("artifact_id") == artifact_id:
                path = artifact.get("path")
                metadata_path = f"{path}_metadata.json"
                
                # Delete the artifact and its metadata
                self.storage_backend.delete(path)
                self.storage_backend.delete(metadata_path)
                
                logger.info(f"Deleted artifact with ID {artifact_id}")
                return True
        
        logger.warning(f"Artifact with ID {artifact_id} not found for deletion")
        return False


class LocalStorageBackend:
    """Storage backend that saves artifacts to the local filesystem."""
    
    def __init__(self, base_path: str):
        """
        Initialize local storage backend.
        
        Args:
            base_path: Base path for storage
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save(self, path: str, data: Any):
        """
        Save data to a path.
        
        Args:
            path: Path to save to
            data: Data to save
        """
        full_path = os.path.join(self.base_path, path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Handle different types of data
        if isinstance(data, (str, int, float, bool, list, dict)):
            with open(full_path, 'w') as f:
                json.dump(data, f)
        else:
            with open(full_path, 'wb') as f:
                pickle.dump(data, f)
    
    def save_metadata(self, path: str, metadata: Dict):
        """
        Save metadata to a path.
        
        Args:
            path: Path to save to
            metadata: Metadata to save
        """
        full_path = os.path.join(self.base_path, path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as f:
            json.dump(metadata, f)
    
    def load(self, path: str) -> Any:
        """
        Load data from a path.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded data
        """
        full_path = os.path.join(self.base_path, path)
        
        try:
            # Try to load as JSON first
            with open(full_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If not JSON, try to load as pickle
            with open(full_path, 'rb') as f:
                return pickle.load(f)
    
    def list_artifacts(self) -> List[Dict]:
        """
        List all artifacts.
        
        Returns:
            List of artifacts with their metadata
        """
        artifacts = []
        
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith("_metadata.json"):
                    metadata_path = os.path.join(root, file)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        artifacts.append(metadata)
        
        return artifacts
    
    def delete(self, path: str):
        """
        Delete a file.
        
        Args:
            path: Path to delete
        """
        full_path = os.path.join(self.base_path, path)
        
        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
