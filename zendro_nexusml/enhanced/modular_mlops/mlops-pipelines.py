# pipelines/base_pipeline.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import datetime
import logging
import os
import json

from ..core.artifact_manager import ArtifactManager
from ..core.experiment_tracker import ExperimentTracker


class BasePipeline(ABC):
    """
    Base class for all ML pipelines.
    
    This provides common functionality for pipelines including:
    - Parameter validation
    - Artifact management
    - Logging
    - Experiment tracking
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.artifact_manager = ArtifactManager()
        self.experiment_tracker = ExperimentTracker()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logger()
        self.start_time = None
        self.end_time = None
        self.experiment_id = None
        self.run_id = None
        
    def _setup_logger(self):
        """Set up logging for the pipeline."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def validate_params(self) -> bool:
        """
        Validate pipeline parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        # Default implementation just checks if required parameters are present
        required_params = self.get_required_params()
        for param in required_params:
            if param not in self.params:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        return True
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """
        Return a list of required parameters for this pipeline.
        
        Returns:
            List of parameter names
        """
        pass
    
    @abstractmethod
    def execute(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline steps.
        
        Args:
            artifacts: Input artifacts
            
        Returns:
            Output artifacts and results
        """
        pass
    
    def run(self, artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the pipeline with tracking and logging.
        
        Args:
            artifacts: Input artifacts
            
        Returns:
            Pipeline execution results
        """
        self.start_time = datetime.datetime.now()
        self.logger.info(f"Starting pipeline: {self.__class__.__name__}")
        
        # Validate parameters
        if not self.validate_params():
            raise ValueError("Invalid pipeline parameters")
            
        # Start experiment tracking
        self.experiment_id = self.experiment_tracker.create_experiment(
            name=self.__class__.__name__,
            description=f"Pipeline run for {self.__class__.__name__}",
            tags=["pipeline", self.__class__.__name__]
        )
        
        self.logger.info(f"Created experiment: {self.experiment_id}")
        
        try:
            # Execute the pipeline
            results = self.execute(artifacts or {})
            
            # Log success
            self.logger.info("Pipeline execution completed successfully")
            
            # Track run results
            metrics = results.get("metrics", {})
            artifact_paths = {
                k: v for k, v in results.items() 
                if isinstance(v, str) and k.startswith("artifact_")
            }
            
            self.run_id = self.experiment_tracker.log_run(
                experiment_id=self.experiment_id,
                params=self.params,
                metrics=metrics,
                artifacts=artifact_paths
            )
            
            self.logger.info(f"Logged run: {self.run_id}")
            
            # Add execution metadata
            self.end_time = datetime.datetime.now()
            results["execution_metadata"] = {
                "pipeline_name": self.__class__.__name__,
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.end_time = datetime.datetime.now()
            raise


# pipelines/example_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from typing import Dict, Any, List, Optional

from ..core.artifact_manager import ArtifactManager
from ..core.model_registry import ModelRegistry
from .base_pipeline import BasePipeline


class ExampleMLPipeline(BasePipeline):
    """
    Example ML Pipeline implementing a simple classification workflow.
    
    This pipeline:
    1. Loads and preprocesses data
    2. Trains a model
    3. Evaluates the model
    4. Saves artifacts
    """
    
    def get_required_params(self) -> List[str]:
        return ["data_path", "target_column", "model_params", "test_size"]
    
    def execute(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline steps."""
        self.logger.info("Starting data loading and preprocessing")
        
        # Load data
        data_path = self.params["data_path"]
        target_column = self.params["target_column"]
        
        # If data is an artifact, use it directly
        if "input_data" in artifacts:
            df = artifacts["input_data"]
        else:
            # Otherwise load from path
            df = pd.read_csv(data_path)
            
        # Basic preprocessing
        df = df.dropna()
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Train/test split
        test_size = self.params["test_size"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Save processed datasets
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data_path = self.artifact_manager.save_artifact(
            artifact=train_data,
            name="train_data",
            artifact_type="dataset",
            metadata={"rows": len(train_data), "columns": train_data.shape[1]}
        )
        
        test_data_path = self.artifact_manager.save_artifact(
            artifact=test_data,
            name="test_data",
            artifact_type="dataset",
            metadata={"rows": len(test_data), "columns": test_data.shape[1]}
        )
        
        self.logger.info("Training model")
        
        # Train model
        model_params = self.params["model_params"]
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        self.logger.info("Evaluating model")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        self.logger.info(f"Model metrics: {metrics}")
        
        # Save model
        model_path = self.artifact_manager.save_artifact(
            artifact=model,
            name="random_forest_model",
            artifact_type="model",
            metadata={
                "model_type": "RandomForestClassifier",
                "parameters": model_params,
                "metrics": metrics,
                "feature_importance": {
                    feature: importance for feature, importance in 
                    zip(X.columns, model.feature_importances_)
                }
            }
        )
        
        # Register the model
        model_registry = ModelRegistry()
        model_version = f"v{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_info = model_registry.register_model(
            name="random_forest_classifier",
            version=model_version,
            model_path=model_path,
            metadata={
                "metrics": metrics,
                "training_params": model_params,
                "dataset": {
                    "path": self.params["data_path"],
                    "target": target_column,
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                }
            },
            description=f"Random Forest Classifier trained on {data_path}",
            tags=["classification", "random_forest"]
        )
        
        # Save results to return
        results = {
            "metrics": metrics,
            "artifact_model": model_path,
            "artifact_train_data": train_data_path,
            "artifact_test_data": test_data_path,
            "model_registry_info": model_info,
            "feature_importance": {
                feature: float(importance) for feature, importance in 
                zip(X.columns, model.feature_importances_)
            }
        }
        
        self.logger.info("Pipeline execution completed")
        return results
