# mlops_framework/core/__init__.py
from .pipeline import Pipeline, PipelineStep
from .artifact import ArtifactStore
from .model_registry import ModelRegistry
from .deployment import DeploymentManager
from .monitoring import Monitor
from .database import Database

# mlops_framework/core/pipeline.py
import inspect
import uuid
import datetime
import logging
from typing import Callable, Dict, Any, List, Optional, Union
from .artifact import ArtifactStore
from .database import Database

class PipelineStep:
    """A reusable pipeline step with defined inputs and outputs."""
    
    def __init__(self, 
                 name: str, 
                 function: Callable, 
                 input_artifacts: Optional[List[str]] = None,
                 output_artifacts: Optional[List[str]] = None,
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize a pipeline step.
        
        Args:
            name: Unique name for the step
            function: The function that implements this step
            input_artifacts: List of artifact names this step requires
            output_artifacts: List of artifact names this step produces
            params: Additional parameters for this step
        """
        self.name = name
        self.function = function
        self.input_artifacts = input_artifacts or []
        self.output_artifacts = output_artifacts or []
        self.params = params or {}
        self.signature = inspect.signature(function)
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline step with given context.
        
        Args:
            context: Dictionary containing input artifacts and parameters
            
        Returns:
            Dictionary of output artifacts produced by this step
        """
        logging.info(f"Executing pipeline step: {self.name}")
        
        # Prepare inputs for the function
        inputs = {}
        for param_name in self.signature.parameters:
            if param_name in context:
                inputs[param_name] = context[param_name]
            elif param_name in self.params:
                inputs[param_name] = self.params[param_name]
                
        # Execute the function
        outputs = self.function(**inputs)
        
        # Ensure output is a dictionary
        if not isinstance(outputs, dict) and len(self.output_artifacts) == 1:
            outputs = {self.output_artifacts[0]: outputs}
            
        return outputs


class Pipeline:
    """A pipeline that chains multiple steps together with automatic data passing."""
    
    def __init__(self, name: str, db: Database = None, artifact_store: ArtifactStore = None):
        """
        Initialize a pipeline.
        
        Args:
            name: Name of the pipeline
            db: Database instance for tracking
            artifact_store: ArtifactStore instance for storing artifacts
        """
        self.name = name
        self.steps = []
        self.db = db or Database()
        self.artifact_store = artifact_store or ArtifactStore(db=self.db)
        
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """
        Add a step to the pipeline.
        
        Args:
            step: PipelineStep instance to add
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self
        
    def run(self, initial_context: Dict[str, Any] = None) -> str:
        """
        Run the pipeline with initial context.
        
        Args:
            initial_context: Initial data to start the pipeline with
            
        Returns:
            Run ID for this pipeline run
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        context = initial_context or {}
        
        # Record pipeline run
        self.db.add_pipeline_run(
            run_id=run_id,
            pipeline_name=self.name,
            timestamp=timestamp,
            status="STARTED"
        )
        
        try:
            # Execute each step
            for step in self.steps:
                # Record step execution
                step_run_id = str(uuid.uuid4())
                self.db.add_step_run(
                    step_run_id=step_run_id,
                    run_id=run_id,
                    step_name=step.name,
                    timestamp=datetime.datetime.now().isoformat(),
                    status="STARTED"
                )
                
                try:
                    # Execute the step
                    step_outputs = step.execute(context)
                    
                    # Store artifacts and update context
                    for artifact_name, artifact_value in step_outputs.items():
                        artifact_id = self.artifact_store.store(
                            name=artifact_name,
                            value=artifact_value,
                            metadata={
                                "run_id": run_id,
                                "step_run_id": step_run_id,
                                "step_name": step.name,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        )
                        
                        # Add to context for next steps
                        context[artifact_name] = artifact_value
                        
                        # Record artifact creation
                        self.db.add_artifact(
                            artifact_id=artifact_id,
                            name=artifact_name,
                            step_run_id=step_run_id,
                            timestamp=datetime.datetime.now().isoformat()
                        )
                    
                    # Update step status
                    self.db.update_step_run(
                        step_run_id=step_run_id,
                        status="COMPLETED",
                        end_timestamp=datetime.datetime.now().isoformat()
                    )
                    
                except Exception as e:
                    # Handle step failure
                    self.db.update_step_run(
                        step_run_id=step_run_id,
                        status="FAILED",
                        end_timestamp=datetime.datetime.now().isoformat(),
                        error=str(e)
                    )
                    raise
            
            # Update pipeline run status
            self.db.update_pipeline_run(
                run_id=run_id,
                status="COMPLETED",
                end_timestamp=datetime.datetime.now().isoformat()
            )
            
            return run_id
            
        except Exception as e:
            # Handle pipeline failure
            self.db.update_pipeline_run(
                run_id=run_id,
                status="FAILED",
                end_timestamp=datetime.datetime.now().isoformat(),
                error=str(e)
            )
            raise

# mlops_framework/core/artifact.py
import os
import pickle
import json
import uuid
import logging
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from .database import Database

class ArtifactStore:
    """Manages storage and retrieval of artifacts with versioning."""
    
    def __init__(self, base_dir: str = "artifacts", db: Database = None):
        """
        Initialize the artifact store.
        
        Args:
            base_dir: Base directory for artifact storage
            db: Database instance for tracking
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.db = db or Database()
        
    def store(self, name: str, value: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Store an artifact.
        
        Args:
            name: Name of the artifact
            value: Artifact value to store
            metadata: Additional metadata to store
            
        Returns:
            Artifact ID
        """
        # Generate artifact ID and version
        artifact_id = str(uuid.uuid4())
        version = self._get_next_version(name)
        
        # Create artifact directory
        artifact_dir = self.base_dir / name / str(version)
        artifact_dir.mkdir(exist_ok=True, parents=True)
        
        # Store the artifact based on its type
        artifact_path = self._store_by_type(value, artifact_dir)
        
        # Store metadata
        metadata = metadata or {}
        metadata.update({
            "artifact_id": artifact_id,
            "name": name,
            "version": version,
            "path": str(artifact_path),
            "type": type(value).__name__
        })
        
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, default=self._json_serializer)
        
        # Record in database
        self.db.add_artifact_version(
            artifact_id=artifact_id,
            name=name,
            version=version,
            path=str(artifact_path),
            metadata=json.dumps(metadata, default=self._json_serializer)
        )
        
        return artifact_id
    
    def load(self, name: str, version: Optional[int] = None) -> Any:
        """
        Load an artifact.
        
        Args:
            name: Name of the artifact
            version: Optional specific version to load (latest if None)
            
        Returns:
            The loaded artifact
        """
        if version is None:
            version = self._get_latest_version(name)
            
        artifact_dir = self.base_dir / name / str(version)
        
        # Load metadata to determine type
        with open(artifact_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load based on type
        return self._load_by_type(metadata["path"], metadata["type"])
    
    def get_versions(self, name: str) -> List[int]:
        """Get all versions of an artifact."""
        versions_dir = self.base_dir / name
        if not versions_dir.exists():
            return []
        
        return sorted([int(d.name) for d in versions_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    def get_metadata(self, name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get metadata for an artifact version."""
        if version is None:
            version = self._get_latest_version(name)
            
        with open(self.base_dir / name / str(version) / "metadata.json", "r") as f:
            return json.load(f)
    
    def _get_next_version(self, name: str) -> int:
        """Get the next version number for an artifact."""
        versions = self.get_versions(name)
        return 1 if not versions else versions[-1] + 1
    
    def _get_latest_version(self, name: str) -> int:
        """Get the latest version number for an artifact."""
        versions = self.get_versions(name)
        if not versions:
            raise ValueError(f"No versions found for artifact {name}")
        return versions[-1]
    
    def _store_by_type(self, value: Any, directory: Path) -> Path:
        """Store artifact based on its type."""
        if isinstance(value, (pd.DataFrame, pd.Series)):
            path = directory / "data.parquet"
            value.to_parquet(path)
        elif isinstance(value, np.ndarray):
            path = directory / "array.npy"
            np.save(path, value)
        elif isinstance(value, (str, int, float, bool, list, dict)):
            path = directory / "data.json"
            with open(path, "w") as f:
                json.dump(value, f, default=self._json_serializer)
        else:
            # Fallback to pickle for complex objects
            path = directory / "data.pkl"
            with open(path, "wb") as f:
                pickle.dump(value, f)
        
        return path
    
    def _load_by_type(self, path: str, type_name: str) -> Any:
        """Load artifact based on its type."""
        path = Path(path)
        
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".npy":
            return np.load(path)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown artifact file type: {path}")
    
    def _json_serializer(self, obj):
        """Handle non-serializable objects for JSON."""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        if isinstance(obj, (Path, bytes)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# mlops_framework/core/model_registry.py
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from .artifact import ArtifactStore
from .database import Database

class ModelRegistry:
    """Model registry for versioning and staging models."""
    
    STAGES = ["development", "staging", "production"]
    
    def __init__(self, artifact_store: ArtifactStore = None, db: Database = None):
        """
        Initialize the model registry.
        
        Args:
            artifact_store: ArtifactStore instance for storing models
            db: Database instance for tracking
        """
        self.artifact_store = artifact_store or ArtifactStore(base_dir="models")
        self.db = db or Database()
    
    def register(self, 
                name: str, 
                model: Any, 
                metadata: Dict[str, Any] = None,
                stage: str = "development") -> str:
        """
        Register a model.
        
        Args:
            name: Name of the model
            model: The model object to register
            metadata: Additional metadata about the model
            stage: Initial stage for this model
            
        Returns:
            Model ID
        """
        if stage not in self.STAGES:
            raise ValueError(f"Stage must be one of {self.STAGES}")
        
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Store the model as an artifact
        metadata = metadata or {}
        metadata["model_id"] = model_id
        metadata["registered_at"] = datetime.datetime.now().isoformat()
        metadata["stage"] = stage
        
        artifact_id = self.artifact_store.store(
            name=name,
            value=model,
            metadata=metadata
        )
        
        # Record in database
        version = self.artifact_store.get_metadata(name)["version"]
        self.db.add_model(
            model_id=model_id,
            name=name,
            version=version,
            artifact_id=artifact_id,
            stage=stage,
            metadata=json.dumps(metadata)
        )
        
        return model_id
    
    def get_model(self, name: str, version: Optional[int] = None, stage: Optional[str] = None) -> Any:
        """
        Get a model by name and version or stage.
        
        Args:
            name: Name of the model
            version: Specific version to retrieve
            stage: Specific stage to retrieve
            
        Returns:
            The model object
        """
        if stage is not None:
            if stage not in self.STAGES:
                raise ValueError(f"Stage must be one of {self.STAGES}")
            
            # Get model from database by stage
            model_info = self.db.get_model_by_stage(name, stage)
            if not model_info:
                raise ValueError(f"No model found with name {name} in stage {stage}")
            
            version = model_info["version"]
        
        # Load from artifact store
        return self.artifact_store.load(name, version)
    
    def promote(self, name: str, version: int, stage: str) -> None:
        """
        Promote a model to a new stage.
        
        Args:
            name: Name of the model
            version: Version of the model
            stage: Target stage
        """
        if stage not in self.STAGES:
            raise ValueError(f"Stage must be one of {self.STAGES}")
        
        # Update stage in database
        self.db.update_model_stage(name, version, stage)
        
        # Update metadata in artifact store
        metadata = self.artifact_store.get_metadata(name, version)
        metadata["stage"] = stage
        metadata["promoted_at"] = datetime.datetime.now().isoformat()
        
        with open(self.artifact_store.base_dir / name / str(version) / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def compare_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models based on their metadata.
        
        Args:
            models: List of model specifications (name, version/stage)
            
        Returns:
            Comparison results
        """
        comparison = {}
        for model_spec in models:
            name = model_spec["name"]
            
            if "version" in model_spec:
                version = model_spec["version"]
            elif "stage" in model_spec:
                model_info = self.db.get_model_by_stage(name, model_spec["stage"])
                if not model_info:
                    raise ValueError(f"No model found with name {name} in stage {model_spec['stage']}")
                version = model_info["version"]
            else:
                raise ValueError("Must specify either version or stage")
            
            # Get metadata for comparison
            metadata = self.artifact_store.get_metadata(name, version)
            comparison[f"{name}_v{version}"] = metadata
            
        return comparison
    
    def list_models(self, name: Optional[str] = None, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models in the registry.
        
        Args:
            name: Filter by name
            stage: Filter by stage
            
        Returns:
            List of model information
        """
        return self.db.list_models(name, stage)

# mlops_framework/core/deployment.py
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from .model_registry import ModelRegistry
from .database import Database

class DeploymentManager:
    """Manages model deployments across environments."""
    
    ENVIRONMENTS = ["dev", "test", "prod"]
    STATUS = ["pending", "deploying", "active", "failed", "stopped"]
    
    def __init__(self, model_registry: ModelRegistry = None, db: Database = None):
        """
        Initialize the deployment manager.
        
        Args:
            model_registry: ModelRegistry instance
            db: Database instance for tracking
        """
        self.model_registry = model_registry or ModelRegistry()
        self.db = db or Database()
    
    def deploy(self, 
              model_name: str, 
              environment: str, 
              version: Optional[int] = None, 
              stage: Optional[str] = None,
              config: Dict[str, Any] = None) -> str:
        """
        Deploy a model to an environment.
        
        Args:
            model_name: Name of the model to deploy
            environment: Target environment
            version: Specific version to deploy
            stage: Specific stage to deploy (alternative to version)
            config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        if environment not in self.ENVIRONMENTS:
            raise ValueError(f"Environment must be one of {self.ENVIRONMENTS}")
        
        if version is None and stage is None:
            raise ValueError("Must provide either version or stage")
        
        if version is None:
            # Get version from stage
            model_info = self.model_registry.db.get_model_by_stage(model_name, stage)
            if not model_info:
                raise ValueError(f"No model found with name {model_name} in stage {stage}")
            version = model_info["version"]
        
        # Generate deployment ID
        deployment_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Record deployment
        self.db.add_deployment(
            deployment_id=deployment_id,
            model_name=model_name,
            model_version=version,
            environment=environment,
            status="pending",
            config=json.dumps(config or {}),
            created_at=timestamp
        )
        
        try:
            # Here you would implement the actual deployment logic
            # This would depend on your infrastructure
            # For example, it might involve copying the model to a serving location,
            # updating a configuration, or calling an external API
            
            # For now, we'll just update the status
            self.db.update_deployment_status(
                deployment_id=deployment_id,
                status="active",
                updated_at=datetime.datetime.now().isoformat()
            )
            
            return deployment_id
            
        except Exception as e:
            # Handle deployment failure
            self.db.update_deployment_status(
                deployment_id=deployment_id,
                status="failed",
                error=str(e),
                updated_at=datetime.datetime.now().isoformat()
            )
            raise
    
    def stop(self, deployment_id: str) -> None:
        """
        Stop a deployment.
        
        Args:
            deployment_id: ID of the deployment to stop
        """
        # Here you would implement the actual stopping logic
        # Again, this depends on your infrastructure
        
        # Update status
        self.db.update_deployment_status(
            deployment_id=deployment_id,
            status="stopped",
            updated_at=datetime.datetime.now().isoformat()
        )
    
    def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get status of a deployment.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Deployment status information
        """
        return self.db.get_deployment(deployment_id)
    
    def list_deployments(self, 
                        model_name: Optional[str] = None, 
                        environment: Optional[str] = None,
                        status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List deployments with optional filtering.
        
        Args:
            model_name: Filter by model name
            environment: Filter by environment
            status: Filter by status
            
        Returns:
            List of deployment information
        """
        return self.db.list_deployments(model_name, environment, status)
    
    def get_deployment_history(self, model_name: str, environment: str) -> List[Dict[str, Any]]:
        """
        Get deployment history for a model in an environment.
        
        Args:
            model_name: Name of the model
            environment: Target environment
            
        Returns:
            List of historical deployments
        """
        return self.db.get_deployment_history(model_name, environment)

# mlops_framework/core/monitoring.py
import json
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from .database import Database
from .deployment import DeploymentManager

class Monitor:
    """Monitors model performance and detects drift."""
    
    def __init__(self, db: Database = None, deployment_manager: DeploymentManager = None):
        """
        Initialize the monitor.
        
        Args:
            db: Database instance for tracking
            deployment_manager: DeploymentManager instance
        """
        self.db = db or Database()
        self.deployment_manager = deployment_manager or DeploymentManager()
    
    def log_prediction(self, 
                      deployment_id: str, 
                      input_data: Any, 
                      prediction: Any,
                      ground_truth: Optional[Any] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Log a prediction for monitoring.
        
        Args:
            deployment_id: ID of the deployment that made the prediction
            input_data: Input data for the prediction
            prediction: Prediction output
            ground_truth: Optional ground truth for evaluation
            metadata: Additional metadata
            
        Returns:
            Prediction ID
        """
        # Convert input_data and prediction to serializable format
        input_serialized = self._serialize_for_storage(input_data)
        prediction_serialized = self._serialize_for_storage(prediction)
        ground_truth_serialized = self._serialize_for_storage(ground_truth) if ground_truth is not None else None
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Record prediction
        self.db.add_prediction(
            prediction_id=prediction_id,
            deployment_id=deployment_id,
            timestamp=timestamp,
            input_data=input_serialized,
            prediction=prediction_serialized,
            ground_truth=ground_truth_serialized,
            metadata=json.dumps(metadata or {})
        )
        
        # Check for drift if we have enough data
        self._check_for_drift(deployment_id)
        
        # Return prediction ID
        return prediction_id
    
    def log_metric(self, 
                  deployment_id: str, 
                  metric_name: str, 
                  value: float,
                  metadata: Dict[str, Any] = None) -> None:
        """
        Log a performance metric.
        
        Args:
            deployment_id: ID of the deployment
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        # Record metric
        self.db.add_metric(
            deployment_id=deployment_id,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.datetime.now().isoformat(),
            metadata=json.dumps(metadata or {})
        )
        
        # Check against thresholds
        self._check_metric_thresholds(deployment_id, metric_name, value)
    
    def set_alert_threshold(self, 
                           metric_name: str, 
                           min_value: Optional[float] = None,
                           max_value: Optional[float] = None,
                           deployment_id: Optional[str] = None) -> None:
        """
        Set alert thresholds for a metric.
        
        Args:
            metric_name: Name of the metric
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            deployment_id: Optional specific deployment ID
        """
        self.db.set_metric_threshold(
            metric_name=metric_name,
            min_value=min_value,
            max_value=max_value,
            deployment_id=deployment_id
        )
    
    def get_metric_history(self, 
                          deployment_id: str, 
                          metric_name: str,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get historical values for a metric.
        
        Args:
            deployment_id: ID of the deployment
            metric_name: Name of the metric
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of historical metric values
        """
        return self.db.get_metric_history(
            deployment_id=deployment_id,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
    
    def calculate_performance(self, 
                             deployment_id: str,
                             start_time: Optional[str] = None,
                             end_time: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate overall performance metrics.
        
        Args:
            deployment_id: ID of the deployment
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary of performance metrics
        """
        # Get predictions with ground truth
        predictions = self.db.get_predictions_with_ground_truth(
            deployment_id=deployment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not predictions:
            return {}
        
        # Calculate metrics based on prediction type
        # This is a simplified implementation - in practice you'd have 
        # more sophisticated logic based on problem type (classification, regression, etc.)
        metrics = {}
        
        # Get sample prediction to determine type
        sample = predictions[0]
        pred_value = json.loads(sample["prediction"])
        truth_value = json.loads(sample["ground_truth"])
        
        # Check if classification or regression
        if isinstance(pred_value, (list, dict)) and isinstance(truth_value, (str, int)):
            # Classification
            accuracy, precision, recall, f1 = self._calculate_classification_metrics(predictions)
            metrics.update({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
        else:
            # Regression
            mae, mse, rmse, r2 = self._calculate_regression_metrics(predictions)
            metrics.update({
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2
            })
        
        # Log all calculated metrics
        for name, value in metrics.items():
            self.log_metric(deployment_id, name, value)
            
        return metrics
    
    def _serialize_for_storage(self, data: Any) -> str:
        """Convert data to JSON-serializable format."""
        if data is None:
            return None
            
        try:
            return json.dumps(data)
        except TypeError:
            # Handle non-serializable types
            if isinstance(data, np.ndarray):
                return json.dumps(data.tolist())
            else:
                return json.dumps(str(data))
    
    def _check_metric_thresholds(self, deployment_id: str, metric_name: str, value: float) -> None:
        """Check if a metric violates thresholds and create alert if needed."""
        # Get thresholds
        threshold = self.db.get_metric_threshold(metric_name, deployment_id)
        
        if not threshold:
            return
        
        # Check against thresholds
        alert_message = None
        if threshold["min_value"] is not None and value < threshold["min_value"]:
            alert_message = f"Metric {metric_name} below threshold: {value} < {threshold['min_value']}"
        elif threshold["max_value"] is not None and value > threshold["max_value"]:
            alert_message = f"Metric {metric_name} above threshold: {value} > {threshold['max_value']}"
        
        if alert_message:
            # Create alert
            self.db.add_alert(
                deployment_id=deployment_id,
                alert_type="metric_threshold",
                message=alert_message,
                timestamp=datetime.datetime.now().isoformat(),
                metadata=json.dumps({
                    "metric_name": metric_name,
                    "value": value,
                    "threshold": threshold
                })
            )
    
    def _check_for_drift(self, deployment_id: str) -> None:
        """Check for data or prediction drift."""
        # Get baseline statistics
        baseline = self.db.get_deployment_baseline(deployment_id)
        
        if not baseline:
            # No baseline, can't check for drift
            return
        
        # Get recent predictions
        recent_predictions = self.db.get_recent_predictions(
            deployment_id=deployment_id,
            