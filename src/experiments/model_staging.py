import time
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from src.utils.config_manager import ConfigManager
from utils.logger import logger


def model_staging(config_file, results):
    """Transitions a registered model to a new stage in MLflow"""
    config = ConfigManager.load_file(config_file)

    # Get MLflow configuration
    tracking_uri = config.get("mlflow_config", {}).get("mlflow_tracking_uri")
    remote_server_uri = config.get("mlflow_config", {}).get("remote_server_uri")
    staging_stage = config.get("mlflow_config", {}).get("stage", {})

    # Set tracking URI - prioritize remote_server_uri if available
    active_uri = remote_server_uri if remote_server_uri else tracking_uri
    if active_uri:
        # logger.info(f"Setting MLflow tracking URI to: {active_uri}")
        mlflow.set_tracking_uri(active_uri)

    # Create MLflow client
    client = MlflowClient()

    model_name = results.get("model_name")
    if not model_name:
        raise ValueError("No model name found in results")

    versions = client.get_latest_versions(model_name, stages=["None"])  # None
    if not versions:
        # logger.warning(f"No model versions found for {model_name} in stage {stage}")
        raise ValueError(f"No versions found for model: {model_name} in stage")

    latest_version = versions[0].version
    # version - latest.version

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage=staging_stage,
        archive_existing_versions=True,  # Archive existing models in the target stage
    )
    print(f"Model_{model_name} version {latest_version} moved to {staging_stage}")

    # Add staging metadata as tags
    client.set_model_version_tag(
        name=model_name,
        version=latest_version,
        key=f"staged_at",
        value=str(int(time.time())),
    )
    client.set_model_version_tag(
        name=model_name,
        version=latest_version,
        key=f"staged_by",
        value=config.get("author", "automated_pipeline"),
    )
    # Get latest version details after staging
    staged_version = client.get_model_version(model_name, latest_version)

    logger.info(f"mlflow tracking model staged to {staging_stage} is completed")

    return {
        "model_name": model_name,
        "version": latest_version,
        "stage": staging_stage,
        "run_id": staged_version.run_id,
        "creation_timestamp": staged_version.creation_timestamp,
        "last_updated_timestamp": int(time.time()),
    }
