import os
import json
import shutil
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersionStatus
from typing import Dict, List, Optional, Tuple, Any, Union
import datetime
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Model registry for managing ML model lifecycle."""
    
    def __init__(self, tracking_uri: str, registry_uri: Optional[str] = None):
        """Initialize model registry.
        
        Args:
            tracking_uri: URI for MLflow tracking server
            registry_uri: URI for MLflow model registry
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri if registry_uri else tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"Initialized model registry with tracking URI: {tracking_uri}")
    
    def register_model(
        self, 
        run_id: str, 
        model_path: str, 
        name: str,
        description: Optional[str] = None
    ) -> str:
        """Register a model from a MLflow run.
        
        Args:
            run_id: MLflow run ID
            model_path: Path to model within the run
            name: Model name in registry
            description: Optional model description
        
        Returns:
            Model version
        """
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model_details = mlflow.register_model(model_uri, name)
            version = model_details.version
            
            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=name,
                    version=version,
                    description=description
                )
            
            logger.info(f"Registered model {name} version {version} from run {run_id}")
            return version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def transition_model_stage(
        self, 
        name: str, 
        version: str, 
        stage: str,
        archive_existing_versions: bool = True
    ) -> None:
        """Transition a model to a different stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions in the target stage
        """
        try:
            # Archive existing versions in the target stage
            if archive_existing_versions and stage in ["Production", "Staging"]:
                current_versions = self.client.search_model_versions(f"name='{name}'")
                for mv in current_versions:
                    if mv.current_stage == stage:
                        logger.info(f"Archiving {name} version {mv.version} from {stage} stage")
                        self.client.transition_model_version_stage(
                            name=name,
                            version=mv.version,
                            stage="Archived"
                        )
            
            # Transition the specified version to the target stage
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Transitioned {name} version {version} to {stage} stage")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise
    
    def get_latest_versions(self, name: str, stages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get latest versions of a model, optionally filtered by stages.
        
        Args:
            name: Model name
            stages: Optional list of stages to filter by
        
        Returns:
            List of model version details
        """
        try:
            versions = self.client.get_latest_versions(name, stages)
            return [
                {
                    "name": mv.name,
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "description": mv.description,
                    "run_id": mv.run_id,
                    "status": mv.status,
                    "creation_timestamp": mv.creation_timestamp
                }
                for mv in versions
            ]
        except Exception as e:
            logger.error(f"Error getting latest versions: {e}")
            raise
    
    def get_model_version(self, name: str, version: str) -> Dict[str, Any]:
        """Get details for a specific model version.
        
        Args:
            name: Model name
            version: Model version
        
        Returns:
            Model version details
        """
        try:
            mv = self.client.get_model_version(name, version)
            return {
                "name": mv.name,
                "version": mv.version,
                "stage": mv.current_stage,
                "description": mv.description,
                "run_id": mv.run_id,
                "status": mv.status,
                "creation_timestamp": mv.creation_timestamp
            }
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            raise
    
    def delete_model_version(self, name: str, version: str) -> None:
        """Delete a specific model version.
        
        Args:
            name: Model name
            version: Model version
        """
        try:
            self.client.delete_model_version(name, version)
            logger.info(f"Deleted {name} version {version}")
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            raise
    
    def delete_registered_model(self, name: str) -> None:
        """Delete a registered model and all its versions.
        
        Args:
            name: Model name
        """
        try:
            self.client.delete_registered_model(name)
            logger.info(f"Deleted registered model {name}")
        except Exception as e:
            logger.error(f"Error deleting registered model: {e}")
            raise
    
    def download_model(
        self, 
        name: str, 
        version: Optional[str] = None, 
        stage: Optional[str] = None,
        dst_path: str = "./downloaded_models"
    ) -> str:
        """Download a model from the registry.
        
        Args:
            name: Model name
            version: Specific version to download (optional)
            stage: Stage to download from (e.g., 'Production') (optional)
            dst_path: Destination directory for downloaded model
        
        Returns:
            Path to downloaded model
        """
        try:
            if version is not None:
                model_uri = f"models:/{name}/{version}"
            elif stage is not None:
                model_uri = f"models:/{name}/{stage}"
            else:
                # Default to latest version
                model_uri = f"models:/{name}/latest"
            
            # Create destination directory with timestamped subfolder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(dst_path, f"{name}_{timestamp}")
            os.makedirs(model_path, exist_ok=True)
            
            # Download the model
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=model_path
            )
            
            logger.info(f"Downloaded model {name} to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def get_model_artifacts(self, name: str, version: str) -> List[Dict[str, Any]]:
        """Get artifacts associated with a model version.
        
        Args:
            name: Model name
            version: Model version
        
        Returns:
            List of artifact details
        """
        try:
            run_id = self.client.get_model_version(name, version).run_id
            artifacts = self.client.list_artifacts(run_id)
            
            return [
                {
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size
                }
                for artifact in artifacts
            ]
        except Exception as e:
            logger.error(f"Error getting model artifacts: {e}")
            raise
    
    def search_models(self, filter_string: str) -> List[Dict[str, Any]]:
        """Search models based on filter criteria.
        
        Args:
            filter_string: Filter string (e.g., "name='iris_model'")
        
        Returns:
            List of model details matching the filter
        """
        try:
            models = self.client.search_registered_models(filter_string)
            return [
                {
                    "name": model.name,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id
                        }
                        for v in model.latest_versions
                    ]
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            raise
    
    def add_model_tags(self, name: str, version: str, tags: Dict[str, Any]) -> None:
        """Add tags to a model version.
        
        Args:
            name: Model name
            version: Model version
            tags: Dictionary of tag keys and values
        """
        try:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, version, key, value)
            
            logger.info(f"Added tags to {name} version {version}: {tags}")
        except Exception as e:
            logger.error(f"Error adding model tags: {e}")
            raise
    
    def wait_for_model_version_ready(
        self, 
        name: str, 
        version: str, 
        timeout_seconds: int = 300
    ) -> bool:
        """Wait for a model version to be in READY status.
        
        Args:
            name: Model name
            version: Model version
            timeout_seconds: Maximum time to wait in seconds
        
        Returns:
            True if model reached READY status, False if timed out
        """
        start_time = datetime.datetime.now()
        
        while (datetime.datetime.now() - start_time).seconds < timeout_seconds:
            model_version = self.client.get_model_version(name, version)
            status = model_version.status
            
            if status == ModelVersionStatus.READY:
                logger.info(f"Model {name} version {version} is now READY")
                return True
            
            if status == ModelVersionStatus.FAILED:
                logger.error(f"Model {name} version {version} FAILED to register")
                return False
            
            logger.info(f"Model {name} version {version} status: {status}, waiting...")
            time.sleep(5)
        
        logger.warning(f"Timed out waiting for model {name} version {version} to be READY")
        return False
