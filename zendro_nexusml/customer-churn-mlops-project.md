# Customer Churn MLOps Project Code

## src/main.py
```python
#!/usr/bin/env python3
"""
Main entry point for the Customer Churn MLOps Pipeline.
This script orchestrates the entire ML workflow from data ingestion to model deployment.
"""
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import prefect
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from src.config import load_config
from src.constants import PIPELINE_REGISTRY
from src.utils.logging_utils import setup_logging
from src.utils.artifact_manager import ArtifactManager
from src.utils.metadata import MetadataTracker

# Import pipeline modules
from src.pipelines.data_ingestion import ingest_data
from src.pipelines.data_validation import validate_data
from src.pipelines.data_preparation import prepare_data
from src.pipelines.feature_engineering import engineer_features
from src.pipelines.training import train_models
from src.pipelines.evaluation import evaluate_models
from src.pipelines.deployment import deploy_best_model
from src.pipelines.retraining import schedule_retraining
from src.pipelines.registry import register_model

logger = logging.getLogger(__name__)


@task(name="Initialize-Run")
def initialize_run(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the pipeline run and create necessary context."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Initializing pipeline run with ID: {run_id}")
    
    # Create artifact manager for this run
    artifact_manager = ArtifactManager(
        base_dir=Path(config["paths"]["artifacts_dir"]) / run_id
    )
    
    # Create metadata tracker
    metadata_tracker = MetadataTracker(
        db_path=Path(config["paths"]["metadata_db"]),
        run_id=run_id
    )
    
    # Save run configuration as artifact
    artifact_manager.save_artifact(
        data=config,
        name="run_config",
        artifact_type="config"
    )
    
    # Record run start in metadata
    metadata_tracker.log_event(
        event_type="run_start",
        data={"config": config}
    )
    
    # Return context for the pipeline
    return {
        "run_id": run_id,
        "config": config,
        "artifact_manager": artifact_manager,
        "metadata_tracker": metadata_tracker
    }


@task(name="Compare-Models")
def compare_models(
    context: Dict[str, Any],
    evaluation_results: Dict[str, Dict[str, float]]
) -> str:
    """Compare trained models and select the best one."""
    logger.info("Comparing model performance metrics")
    
    # Get the primary metric for model selection
    primary_metric = context["config"]["model"]["primary_metric"]
    is_higher_better = context["config"]["model"]["higher_is_better"]
    
    # Determine the best model based on the primary metric
    best_model_id = None
    best_metric_value = float('-inf') if is_higher_better else float('inf')
    
    for model_id, metrics in evaluation_results.items():
        current_value = metrics[primary_metric]
        
        if (is_higher_better and current_value > best_metric_value) or \
           (not is_higher_better and current_value < best_metric_value):
            best_metric_value = current_value
            best_model_id = model_id
    
    # Log the selection
    logger.info(f"Best model selected: {best_model_id} with {primary_metric} = {best_metric_value}")
    
    # Save comparison results as artifact
    context["artifact_manager"].save_artifact(
        data=evaluation_results,
        name="model_comparison",
        artifact_type="metrics"
    )
    
    # Record in metadata
    context["metadata_tracker"].log_event(
        event_type="model_selection",
        data={
            "best_model_id": best_model_id,
            "primary_metric": primary_metric,
            "metric_value": best_metric_value,
            "all_metrics": evaluation_results
        }
    )
    
    return best_model_id


@flow(name="Customer-Churn-Prediction-Pipeline", task_runner=SequentialTaskRunner())
def run_pipeline(config_path: str = "configs/params.yaml") -> None:
    """
    Execute the end-to-end ML pipeline for customer churn prediction.
    
    Args:
        config_path: Path to the configuration file
    """
    # Setup logging
    setup_logging()
    logger.info("Starting Customer Churn Prediction Pipeline")
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize the run and get context
    context = initialize_run(config)
    run_id = context["run_id"]
    
    try:
        # Data ingestion
        raw_data_path = ingest_data(context)
        
        # Data validation
        validation_results = validate_data(context, raw_data_path)
        
        if not validation_results["passed"]:
            logger.error("Data validation failed. Aborting pipeline.")
            context["metadata_tracker"].log_event(
                event_type="run_failed",
                data={"reason": "data_validation_failure", "details": validation_results}
            )
            return
        
        # Data preparation
        prepared_data_path = prepare_data(context, raw_data_path)
        
        # Feature engineering
        feature_data_paths = engineer_features(context, prepared_data_path)
        
        # Model training
        trained_models = train_models(context, feature_data_paths)
        
        # Model evaluation
        evaluation_results = evaluate_models(context, trained_models, feature_data_paths)
        
        # Model comparison and selection
        best_model_id = compare_models(context, evaluation_results)
        
        # Register the best model
        model_registry_info = register_model(
            context, 
            model_id=best_model_id,
            metrics=evaluation_results[best_model_id]
        )
        
        # Deploy the best model
        deployment_result = deploy_best_model(
            context,
            model_registry_info=model_registry_info
        )
        
        # Schedule retraining if needed
        if config["pipeline"].get("schedule_retraining", False):
            retraining_job = schedule_retraining(context)
            logger.info(f"Retraining scheduled with job ID: {retraining_job}")
        
        # Log successful completion
        logger.info(f"Pipeline run {run_id} completed successfully")
        context["metadata_tracker"].log_event(
            event_type="run_completed",
            data={"status": "success"}
        )
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {str(e)}")
        context["metadata_tracker"].log_event(
            event_type="run_failed",
            data={"reason": "exception", "error": str(e)}
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn MLOps Pipeline")
    parser.add_argument(
        "--config", 
        default="configs/params.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    run_pipeline(config_path=args.config)
```

## src/config.py
```python
"""
Configuration loader for the MLOps pipeline.
Handles loading and validating configuration from YAML files.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

class PipelineConfig(BaseModel):
    """Validate pipeline configuration using Pydantic."""
    # Paths
    paths: Dict[str, str]
    
    # Data settings
    data: Dict[str, Any]
    
    # Feature settings
    features: Dict[str, Any]
    
    # Model settings
    model: Dict[str, Any]
    
    # Evaluation settings
    evaluation: Dict[str, Any]
    
    # Pipeline settings
    pipeline: Dict[str, Any]
    
    # Infrastructure settings
    infrastructure: Dict[str, Any]
    
    # Monitoring settings
    monitoring: Optional[Dict[str, Any]]

    @validator('paths')
    def check_required_paths(cls, paths):
        """Ensure all required paths are defined."""
        required_paths = [
            'raw_data_dir', 
            'processed_data_dir', 
            'models_dir',
            'artifacts_dir',
            'metadata_db'
        ]
        
        for path in required_paths:
            if path not in paths:
                raise ValueError(f"Required path '{path}' is missing from configuration")
        
        return paths


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and validate it.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validated_config = PipelineConfig(**config).dict()
        
        # Resolve any relative paths
        for key, path in validated_config["paths"].items():
            if not os.path.isabs(path):
                validated_config["paths"][key] = os.path.abspath(
                    os.path.join(os.path.dirname(config_file), path)
                )
                
        logger.info(f"Configuration loaded successfully from {config_path}")
        return validated_config
    
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise
```

## src/constants.py
```python
"""
Constants for the MLOps pipeline.
"""
from enum import Enum, auto
from typing import Dict, List

# Pipeline stages registry
PIPELINE_REGISTRY = {
    "data_ingestion": "data_ingestion",
    "data_validation": "data_validation",
    "data_preparation": "data_preparation",
    "feature_engineering": "feature_engineering",
    "training": "training",
    "evaluation": "evaluation",
    "deployment": "deployment",
    "monitoring": "monitoring",
    "retraining": "retraining"
}

# Model types
class ModelType(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XG_BOOST = "xgboost"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"

# Feature types
class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DATETIME = "datetime"
    TEXT = "text"

# Model evaluation metrics
CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]

# Environment types
class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

# Default model hyperparameter search spaces
DEFAULT_HYPERPARAMETER_SPACES = {
    ModelType.LOGISTIC_REGRESSION.value: {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [100, 200, 300]
    },
    ModelType.RANDOM_FOREST.value: {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    },
    ModelType.GRADIENT_BOOSTING.value: {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 0.9, 1.0]
    },
    ModelType.XG_BOOST.value: {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2]
    }
}

# Default data schema
DEFAULT_DATA_SCHEMA = {
    "customer_id": {"type": FeatureType.CATEGORICAL.value, "required": True},
    "tenure": {"type": FeatureType.NUMERICAL.value, "required": True},
    "contract_type": {"type": FeatureType.CATEGORICAL.value, "required": True},
    "monthly_charges": {"type": FeatureType.NUMERICAL.value, "required": True},
    "total_charges": {"type": FeatureType.NUMERICAL.value, "required": True},
    "churn": {"type": FeatureType.CATEGORICAL.value, "required": True, "target": True}
}

# API rate limits (requests per minute)
API_RATE_LIMITS = {
    EnvironmentType.DEVELOPMENT.value: 100,
    EnvironmentType.STAGING.value: 300,
    EnvironmentType.PRODUCTION.value: 1000
}

# Default monitoring thresholds
DEFAULT_MONITORING_THRESHOLDS = {
    "drift_threshold": 0.1,
    "performance_degradation_threshold": 0.05,
    "data_quality_threshold": 0.9,
    "missing_values_threshold": 0.1
}
```

## src/utils/artifact_manager.py
```python
"""
Artifact management utilities for the MLOps pipeline.
Handles saving, loading, and tracking ML artifacts.
"""
import os
import json
import pickle
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
import shutil
import hashlib
import uuid

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    Handles storage and retrieval of artifacts throughout the ML pipeline.
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the ArtifactManager.
        
        Args:
            base_dir: Base directory to store artifacts
        """
        self.base_dir = Path(base_dir)
        self._ensure_directories()
        self.artifact_registry = {}
    
    def _ensure_directories(self) -> None:
        """Create necessary directories for artifacts."""
        directories = [
            self.base_dir,
            self.base_dir / "data",
            self.base_dir / "models",
            self.base_dir / "metrics",
            self.base_dir / "config",
            self.base_dir / "metadata",
            self.base_dir / "plots"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_artifact_path(self, name: str, artifact_type: str) -> Path:
        """
        Get the appropriate path for an artifact.
        
        Args:
            name: Name of the artifact
            artifact_type: Type of the artifact
            
        Returns:
            Path where the artifact should be stored
        """
        type_dir_map = {
            "data": "data",
            "model": "models",
            "metrics": "metrics",
            "config": "config",
            "metadata": "metadata",
            "plot": "plots"
        }
        
        if artifact_type not in type_dir_map:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        return self.base_dir / type_dir_map[artifact_type] / name
    
    def save_artifact(
        self, 
        data: Any, 
        name: str, 
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save an artifact to storage.
        
        Args:
            data: The artifact data to save
            name: Name of the artifact
            artifact_type: Type of the artifact
            metadata: Additional metadata to store with the artifact
            
        Returns:
            Path to the saved artifact
        """
        artifact_path = self._get_artifact_path(name, artifact_type)
        
        # Generate a unique ID for the artifact
        artifact_id = str(uuid.uuid4())
        
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            file_path = f"{artifact_path}.parquet"
            data.to_parquet(file_path, index=False)
        
        elif isinstance(data, (dict, list)):
            file_path = f"{artifact_path}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif isinstance(data, (np.ndarray)):
            file_path = f"{artifact_path}.npy"
            np.save(file_path, data)
        
        else:
            file_path = f"{artifact_path}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        
        # Calculate hash of the file for integrity tracking
        file_hash = self._calculate_file_hash(file_path)
        
        # Record artifact in registry
        artifact_info = {
            "id": artifact_id,
            "name": name,
            "type": artifact_type,
            "path": str(file_path),
            "hash": file_hash,
            "created_at": pd.Timestamp.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save artifact registry with the updated info
        registry_path = self.base_dir / "artifact_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.artifact_registry = json.load(f)
        
        self.artifact_registry[artifact_id] = artifact_info
        
        with open(registry_path, 'w') as f:
            json.dump(self.artifact_registry, f, indent=2, default=str)
        
        logger.info(f"Artifact '{name}' of type '{artifact_type}' saved to {file_path}")
        return artifact_id
    
    def load_artifact(self, artifact_id: str) -> Any:
        """
        Load an artifact from storage by its ID.
        
        Args:
            artifact_id: ID of the artifact to load
            
        Returns:
            The loaded artifact data
        """
        registry_path = self.base_dir / "artifact_registry.json"
        
        if not registry_path.exists():
            raise FileNotFoundError("Artifact registry not found")
        
        with open(registry_path, 'r') as f:
            self.artifact_registry = json.load(f)
        
        if artifact_id not in self.artifact_registry:
            raise ValueError(f"Artifact with ID {artifact_id} not found in registry")
        
        artifact_info = self.artifact_registry[artifact_id]
        file_path = artifact_info["path"]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        
        # Verify file integrity
        current_hash = self._calculate_file_hash(file_path)
        if current_hash != artifact_info["hash"]:
            logger.warning(f"Artifact hash mismatch for {artifact_id}. File may be corrupted.")
        
        # Load based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.parquet':
            return pd.read_parquet(file_path)
        
        elif file_ext == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif file_ext == '.npy':
            return np.load(file_path)
        
        elif file_ext == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    
    def load_artifact_by_name(self, name: str, artifact_type: str) -> Any:
        """
        Load the most recent artifact with the given name and type.
        
        Args:
            name: Name of the artifact
            artifact_type: Type of the artifact
            
        Returns:
            The loaded artifact data
        """
        registry_path = self.base_dir / "artifact_registry.json"
        
        if not registry_path.exists():
            raise FileNotFoundError("Artifact registry not found")
        
        with open(registry_path, 'r') as f:
            self.artifact_registry = json.load(f)
        
        # Find matching artifacts and sort by creation time (most recent first)
        matching_artifacts = [
            a for a in self.artifact_registry.values()
            if a["name"] == name and a["type"] == artifact_type
        ]
        
        if not matching_artifacts:
            raise ValueError(f"No artifacts found with name '{name}' and type '{artifact_type}'")
        
        # Sort by creation time and get the most recent
        most_recent = sorted(
            matching_artifacts, 
            key=lambda x: x["created_at"], 
            reverse=True
        )[0]
        
        return self.load_artifact(most_recent["id"])
    
    def list_artifacts(self, artifact_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List all artifacts, optionally filtered by type.
        
        Args:
            artifact_type: Optional filter for artifact type
            
        Returns:
            Dictionary of artifact information
        """
        registry_path = self.base_dir / "artifact_registry.json"
        
        if not registry_path.exists():
            return {}
        
        with open(registry_path, 'r') as f:
            self.artifact_registry = json.load(f)
        
        if artifact_type:
            return {
                id: info for id, info in self.artifact_registry.items()
                if info["type"] == artifact_type
            }
        
        return self.artifact_registry
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hex digest of the file hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
```

## src/utils/metadata.py
```python
"""
Metadata tracking for the MLOps pipeline.
Records and manages metadata about pipeline runs, models, and artifacts.
"""
import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataTracker:
    """
    Track and store metadata about pipeline runs and ML artifacts.
    Uses SQLite database for persistent storage.
    """
    
    def __init__(self, db_path: Union[str, Path], run_id: str):
        """
        Initialize the MetadataTracker.
        
        Args:
            db_path: Path to the SQLite database file
            run_id: ID of the current pipeline run
        """
        self.db_path = Path(db_path)
        self.run_id = run_id
        self._ensure_db_exists()
    
    def _ensure_db_exists(self) -> None:
        """Create the database and necessary tables if they don't exist."""
        # Create parent directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database and create tables
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Create runs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT,
                config TEXT
            )
            ''')
            
            # Create events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
            ''')
            
            # Create models table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model_type TEXT NOT NULL,
                metrics TEXT,
                parameters TEXT,
                artifact_path TEXT,
                status TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
            ''')
            
            # Create deployments table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                deployed_at TEXT NOT NULL,
                environment TEXT NOT NULL,
                status TEXT,
                metadata TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
            ''')
            
            # Insert run record for this pipeline run
            cursor.execute(
                "INSERT INTO runs (run_id, start_time, status, config) VALUES (?, ?, ?, ?)",
                (self.run_id, datetime.now().isoformat(), "RUNNING", "{}")
            )
            
            conn.commit()
            
        logger.info(f"Metadata database initialized at {self.db_path}")
    
    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> int:
        """
        Log an event for the current pipeline run.
        
        Args:
            event_type: Type of the event
            data: Additional data for the event
            
        Returns:
            ID of the created event record
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO events (run_id, timestamp, event_type, data) VALUES (?, ?, ?, ?)",
                (
                    self.run_id,
                    datetime.now().isoformat(),
                    event_type,
                    json.dumps(data) if data else None
                )
            )
            
            conn.commit()
            event_id = cursor.lastrowid
        
        logger.debug(f"Logged event '{event_type}' for run {self.run_id}")
        return event_id
    
    def update_run_status(self, status: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of the current pipeline run.
        
        Args:
            status: New status for the run
            config: Updated configuration for the run
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            if status.upper() in ["COMPLETED", "FAILED"]:
                # Set end time if the run is finishing
                if config:
                    cursor.execute(
                        "UPDATE runs SET status = ?, end_time = ?, config = ? WHERE run_id = ?",
                        (status.upper(), datetime.now().isoformat(), json.dumps(config), self.run_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?",
                        (status.upper(), datetime.now().isoformat(), self.run_id)
                    )
            else:
                # Just update status without setting end time
                if config:
                    cursor.execute(
                        "UPDATE runs SET status = ?, config = ? WHERE run_id = ?",
                        (status.upper(), json.dumps(config), self.run_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE runs SET status = ? WHERE run_id = ?",
                        (status.upper(), self.run_id)
                    )
            
            conn.commit()
        
        logger.info(f"Updated run {self.run_id} status to {status}")
    
    def register_model(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        artifact_path: str,
        status: str = "REGISTERED"
    ) -> None:
        """
        Register a trained model in the metadata database.
        
        Args:
            model_id: Unique ID for the model
            model_type: Type of the model (e.g., "RandomForest", "XGBoost")
            metrics: Performance metrics for the model
            parameters: Hyperparameters used for training
            artifact_path: Path to the model artifact
            status: Status of the model
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO models 
                (model_id, run_id, created_at, model_type, metrics, parameters, artifact_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    self.run_id,
                    datetime.now().isoformat(),
                    model_type,
                    json.dumps(metrics),
                    json.dumps(parameters),
                    artifact_path,
                    status.upper()
                )
            )
            
            conn.commit()
        
        logger.info(f"Registered model {model_id} of type {model_type}")
    
    def register_deployment(
        self,
        deployment_id: str,
        model_id: str,
        environment: str,
        status: str = "DEPLOYED",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a model deployment in the metadata database.
        
        Args:
            deployment_id: Unique ID for the deployment
            model_id: ID of the deployed model
            environment: