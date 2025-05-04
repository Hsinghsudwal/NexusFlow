

This module provides interfaces to MLflow's model registry for managing model lifecycle
in an enterprise setting.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Enterprise Model Registry using MLflow."""
    
    def __init__(self, tracking_uri: Optional[str] = None, 
                 registry_uri: Optional[str] = None):
        """
        Initialize the ModelRegistry with MLflow configuration.
        
        Args:
            tracking_uri: URI for MLflow tracking server
            registry_uri: URI for MLflow registry server
        """
        # Set MLflow tracking and registry URIs from args or environment variables
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient()
        logger.info(f"ModelRegistry initialized with tracking URI: {self.tracking_uri} "
                   f"and registry URI: {self.registry_uri}")
    
    def register_model(self, run_id: str, model_path: str, name: str) -> Any:
        """
        Register a model from a given run to the model registry.
        
        Args:
            run_id: MLflow run ID containing the model
            model_path: Path to the model in the MLflow run
            name: Name to register the model under
        
        Returns:
            ModelVersion: The registered model version
        """
        logger.info(f"Registering model {name} from run {run_id}")
        model_uri = f"runs:/{run_id}/{model_path}"
        model_version = mlflow.register_model(model_uri, name)
        logger.info(f"Registered model version: {model_version.version}")
        return model_version
    
    def promote_model(self, name: str, version: int, stage: str, 
                      description: Optional[str] = None) -> Any:
        """
        Promote a model version to a new stage.
        
        Args:
            name: Name of the registered model
            version: Version number to promote
            stage: Target stage (Staging, Production, Archived)
            description: Optional description of why this promotion is happening
        
        Returns:
            ModelVersion: The updated model version
        """
        logger.info(f"Promoting model {name} version {version} to {stage}")
        # Add timestamp and stage transition to description
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_description = f"[{timestamp}] Promoted to {stage}."
        if description:
            full_description += f" Reason: {description}"
            
        # Update model stage
        model_version = self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production")
        )
        
        # Update description to include promotion history
        current_desc = model_version.description or ""
        if current_desc:
            full_description = f"{current_desc}\n{full_description}"
        
        self.client.update_model_version(
            name=name,
            version=version,
            description=full_description
        )
        
        logger.info(f"Model {name} version {version} promoted to {stage}")
        return self.client.get_model_version(name, version)
    
    def get_latest_versions(self, name: str, stages: Optional[List[str]] = None) -> List[Any]:
        """
        Get the latest versions of a model, optionally filtering by stage.
        
        Args:
            name: Name of the registered model
            stages: Optional list of stages to filter by
        
        Returns:
            List[ModelVersion]: List of model versions
        """
        logger.info(f"Getting latest versions of model {name}" + 
                   (f" in stages {stages}" if stages else ""))
        return self.client.get_latest_versions(name, stages)
    
    def get_model_version(self, name: str, version: int) -> Any:
        """
        Get a specific model version.
        
        Args:
            name: Name of the registered model
            version: Version number to retrieve
        
        Returns:
            ModelVersion: The requested model version
        """
        logger.info(f"Getting model {name} version {version}")
        return self.client.get_model_version(name, version)
    
    def add_model_tag(self, name: str, version: int, key: str, value: str) -> None:
        """
        Add a tag to a model version.
        
        Args:
            name: Name of the registered model
            version: Version number to tag
            key: Tag key
            value: Tag value
        """
        logger.info(f"Adding tag {key}={value} to model {name} version {version}")
        self.client.set_model_version_tag(name, version, key, value)
    
    def log_model_approval(self, name: str, version: int, approved: bool, 
                          approved_by: str, reason: Optional[str] = None) -> None:
        """
        Log a model approval decision as tags.
        
        Args:
            name: Name of the registered model
            version: Version number 
            approved: Whether the model was approved
            approved_by: Name of the approver
            reason: Optional reason for the decision
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.add_model_tag(name, version, "approval_status", "approved" if approved else "rejected")
        self.add_model_tag(name, version, "approved_by", approved_by)
        self.add_model_tag(name, version, "approval_timestamp", timestamp)
        
        if reason:
            self.add_model_tag(name, version, "approval_reason", reason)
            
        logger.info(f"Logged model {name} version {version} approval: {approved} by {approved_by}")
    
    def create_registered_model(self, name: str, description: Optional[str] = None, 
                               tags: Optional[Dict[str, str]] = None) -> Any:
        """
        Create a new registered model.
        
        Args:
            name: Name for the new model
            description: Optional description
            tags: Optional tags as key-value pairs
        
        Returns:
            RegisteredModel: The created model
        """
        logger.info(f"Creating registered model {name}")
        return self.client.create_registered_model(name, description, tags)
    
    def get_deployment_url(self, name: str, version: int, stage: str = "Production") -> str:
        """
        Construct the URL for accessing a deployed model.
        
        Args:
            name: Name of the registered model
            version: Version number
            stage: Deployment stage (default: Production)
            
        Returns:
            str: URL for the deployed model
        """
        # This is a template - actual implementation depends on your deployment infrastructure
        base_url = os.getenv("MODEL_SERVING_BASE_URL", "http://localhost:8000")
        model_endpoint = f"/models/{name}/versions/{version}"
        
        logger.info(f"Generated deployment URL for model {name} version {version}: {base_url}{model_endpoint}")
        return f"{base_url}{model_endpoint}"

    def archive_models_except_latest(self, name: str, keep_stages: List[str] = ["Production", "Staging"]) -> None:
        """
        Archive all model versions except the latest ones in specified stages.
        
        Args:
            name: Name of the registered model
            keep_stages: Stages where the latest versions should be kept
        """
        # Get all versions
        all_versions = self.client.search_model_versions(f"name='{name}'")
        
        # Get latest versions in each stage we want to keep
        latest_versions = {}
        for stage in keep_stages:
            stage_versions = [v for v in all_versions if v.current_stage == stage]
            if stage_versions:
                # Sort by version number (highest first)
                stage_versions.sort(key=lambda x: int(x.version), reverse=True)
                latest_versions[stage] = stage_versions[0].version
        
        # Archive all versions except the ones we want to keep
        for version in all_versions:
            if (version.current_stage not in ["Archived"] and 
                (version.current_stage not in keep_stages or 
                 version.version != latest_versions.get(version.current_stage))):
                logger.info(f"Archiving model {name} version {version.version} from {version.current_stage}")
                self.client.transition_model_version_stage(
                    name=name,
                    version=version.version,
                    stage="Archived"
                )