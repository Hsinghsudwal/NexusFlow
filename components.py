# custom_mlops/components.py
"""Components for the custom MLOps framework."""

import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import mlflow
import tempfile
import datetime

from .core import Step, Artifact, ArtifactStore, Pipeline

logger = logging.getLogger(__name__)

# Data components
class DataLoader(Step):
    """Load data from various sources."""
    
    def __init__(self, name: str = "DataLoader", source_type: str = "csv", 
                 source_path: Optional[str] = None, params: Dict[str, Any] = None):
        super().__init__(name)
        self.source_type = source_type
        self.source_path = source_path
        self.params = params or {}
    
    def run(self, source_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Load data from the specified source."""
        path = source_path or self.source_path
        if not path:
            raise ValueError("Source path not provided")
        
        if self.source_type == "csv":
            data = pd.read_csv(path, **self.params)
        elif self.source_type == "parquet":
            data = pd.read_parquet(path, **self.params)
        elif self.source_type == "json":
            data = pd.read_json(path, **self.params)
        elif self.source_type == "pickle":
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        
        return {"data": data}


class DataSplitter(Step):
    """Split data into train and test sets."""
    
    def __init__(self, name: str = "DataSplitter", test_size: float = 0.2, 
                 random_state: Optional[int] = None, stratify_col: Optional[str] = None):
        super().__init__(name)
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_col = stratify_col
    
    def run(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        if self.stratify_col and self.stratify_col in data.columns:
            stratify = data[self.stratify_col]
        else:
            stratify = None
        
        X = data.drop(columns=[self.stratify_col]) if self.stratify_col else data
        
        train_data, test_data = train_test_split(
            X, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        return {
            "train_data": train_data,
            "test_data": test_data
        }


class FeatureEngineeringStep(Step):
    """Apply feature engineering transformations."""
    
    def __init__(self, name: str = "FeatureEngineering", 
                 transformations: List[Tuple[str, Callable]] = None):
        super().__init__(name)
        self.transformations = transformations or []
    
    def add_transformation(self, name: str, func: Callable):
        """Add a transformation function."""
        self.transformations.append((name, func))
        return self
    
    def run(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Apply transformations to the data."""
        result = data.copy()
        
        for name, func in self.transformations:
            logger.info(f"Applying transformation: {name}")
            result = func(result)
        
        return {"data": result}


# Model components
class ModelTrainer(Step):
    """Train a machine learning model."""
    
    def __init__(self, name: str = "ModelTrainer", 
                 model: Optional[BaseEstimator] = None,
                 feature_cols: Optional[List[str]] = None,
                 target_col: Optional[str] = None,
                 params: Dict[str, Any] = None):
        super().__init__(name)
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.params = params or {}
    
    def run(self, train_data: pd.DataFrame, 
            feature_cols: Optional[List[str]] = None, 
            target_col: Optional[str] = None, 
            model: Optional[BaseEstimator] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the model on the provided data."""
        features = feature_cols or self.feature_cols
        target = target_col or self.target_col
        
        if not features or not target:
            raise ValueError("Feature columns and target column must be specified")
        
        X = train_data[features]
        y = train_data[target]
        
        model_to_train = model or self.model
        if not model_to_train:
            raise ValueError("Model not provided")
        
        # Clone the model to avoid modifying the original
        from sklearn.base import clone
        model_instance = clone(model_to_train)
        
        # Set parameters if provided
        if self.params:
            model_instance.set_params(**self.params)
        
        # Train the model
        model_instance.fit(X, y)
        
        return {
            "model": model_instance,
            "feature_cols": features,
            "target_col": target
        }


class ModelEvaluator(Step):
    """Evaluate a trained model."""
    
    def __init__(self, name: str = "ModelEvaluator", metrics: List[Tuple[str, Callable]] = None):
        super().__init__(name)
        self.metrics = metrics or []
        
        # Add default metrics if none provided
        if not self.metrics:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
            
            self.metrics = [
                ("accuracy", accuracy_score),
                ("precision", lambda y, y_pred: precision_score(y, y_pred, average='weighted')),
                ("recall", lambda y, y_pred: recall_score(y, y_pred, average='weighted')),
                ("f1", lambda y, y_pred: f1_score(y, y_pred, average='weighted')),
                ("mse", mean_squared_error),
                ("r2", r2_score)
            ]
    
    def add_metric(self, name: str, metric_func: Callable):
        """Add a custom evaluation metric."""
        self.metrics.append((name, metric_func))
        return self
    
    def run(self, model: BaseEstimator, test_data: pd.DataFrame, 
            feature_cols: List[str], target_col: str, **kwargs) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
        
        # Calculate metrics
        evaluation_results = {}
        for metric_name, metric_func in self.metrics:
            try:
                score = metric_func(y_test, y_pred)
                evaluation_results[metric_name] = score
                logger.info(f"{metric_name}: {score}")
            except Exception as e:
                logger.warning(f"Could not compute {metric_name}: {e}")
        
        return {
            "predictions": y_pred,
            "metrics": evaluation_results
        }


# Tracking and deployment components
class MLflowTracker(Step):
    """Track experiments with MLflow."""
    
    def __init__(self, name: str = "MLflowTracker", 
                 experiment_name: str = "Default Experiment",
                 tracking_uri: Optional[str] = None):
        super().__init__(name)
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    
    def run(self, model: BaseEstimator, metrics: Dict[str, float], 
            feature_cols: List[str], params: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Log model, parameters, and metrics to MLflow."""
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Start run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log model parameters
            try:
                model_params = model.get_params()
                mlflow.log_params(model_params)
            except:
                logger.warning("Could not log model parameters")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            try:
                mlflow.sklearn.log_model(model, "model")
                
                # Log feature info
                feature_info = {
                    "feature_columns": feature_cols,
                    "num_features": len(feature_cols)
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                    json.dump(feature_info, f)
                    f.flush()
                    mlflow.log_artifact(f.name, "feature_info")
                
            except Exception as e:
                logger.error(f"Error logging model: {e}")
        
        return {
            "run_id": run_id,
            "experiment_name": self.experiment_name
        }


class ModelSerializer(Step):
    """Serialize a trained model."""
    
    def __init__(self, name: str = "ModelSerializer", 
                 output_dir: str = "models",
                 format: str = "pickle"):
        super().__init__(name)
        self.output_dir = output_dir
        self.format = format
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, model: BaseEstimator, metrics: Dict[str, float] = None, 
            feature_cols: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Serialize the model to disk."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}"
        
        model_path = os.path.join(self.output_dir, model_filename)
        
        # Save the model
        if self.format == "pickle":
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(model, f)
            file_path = f"{model_path}.pkl"
        elif self.format == "joblib":
            from joblib import dump
            dump(model, f"{model_path}.joblib")
            file_path = f"{model_path}.joblib"
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "metrics": metrics or {},
            "feature_columns": feature_cols or [],
            "model_type": type(model).__name__,
            "format": self.format
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "model_path": file_path,
            "metadata_path": f"{model_path}_metadata.json"
        }