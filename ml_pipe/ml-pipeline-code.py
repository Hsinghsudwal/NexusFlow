# pipeline.py - ML training and deployment pipeline

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
import great_expectations as ge
from feast import FeatureStore
import optuna
import yaml

class MLPipeline:
    """End-to-end ML pipeline with MLOps best practices."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
        # Initialize feature store
        self.feature_store = FeatureStore(repo_path=self.config["feature_store"]["repo_path"])
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    def extract_data(self) -> pd.DataFrame:
        """Extract data from feature store."""
        entity_df = pd.read_parquet(self.config["data"]["entity_path"])
        
        # Get features from feature store
        features = self.feature_store.get_historical_features(
            entity_df=entity_df,
            feature_refs=self.config["feature_store"]["feature_refs"],
        ).to_df()
        
        print(f"Extracted {len(features)} records with {len(features.columns)} features")
        return features
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality using Great Expectations."""
        data_context = ge.data_context.DataContext(
            context_root_dir=self.config["data_validation"]["ge_root_dir"]
        )
        
        batch = ge.dataset.PandasDataset(df)
        
        # Run validation suite
        results = batch.validate(
            expectation_suite_name=self.config["data_validation"]["expectation_suite"]
        )
        
        # Log validation results
        validation_passed = results["success"]
        print(f"Data validation {'passed' if validation_passed else 'failed'}")
        
        if not validation_passed:
            failed_expectations = [
                exp for exp in results["results"] if not exp["success"]
            ]
            print(f"Failed validations: {len(failed_expectations)}")
            for exp in failed_expectations[:5]:  # Show first 5 failures
                print(f"- {exp['expectation_config']['expectation_type']}")
        
        return validation_passed
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data for training."""
        # Split features and target
        X = df.drop(columns=[self.config["data"]["target_column"]])
        y = df[self.config["data"]["target_column"]]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config["training"]["test_size"],
            random_state=self.config["training"]["random_state"],
            stratify=y if self.config["training"]["stratify"] else None
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
            }
            
            model = RandomForestClassifier(**params, random_state=self.config["training"]["random_state"])
            model.fit(X_train, y_train)
            
            # Use cross-validation score as optimization metric
            return model.score(X_train, y_train)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, 
            n_trials=self.config["hyperparameter_tuning"]["n_trials"]
        )
        
        print(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, params: Dict) -> RandomForestClassifier:
        """Train model with optimized hyperparameters."""
        model = RandomForestClassifier(
            **params,
            random_state=self.config["training"]["random_state"]
        )
        
        # Train the model
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted")
        }
        
        print(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def log_model(self, model: RandomForestClassifier, X_test: np.ndarray, metrics: Dict) -> str:
        """Log model and metrics to MLflow."""
        with mlflow.start_run(run_name=f"run_{self.run_id}"):
            # Log hyperparameters
            for key, value in model.get_params().items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log feature importance
            feature_importance = pd.DataFrame(
                model.feature_importances_,
                index=X_test.columns,
                columns=["importance"]
            ).sort_values("importance", ascending=False)
            
            # Save feature importance
            importance_path = f"feature_importance_{self.run_id}.csv"
            feature_importance.to_csv(importance_path)
            mlflow.log_artifact(importance_path)
            
            # Log model with signature
            signature = infer_signature(X_test, model.predict(X_test))
            model_path = mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                registered_model_name=self.config["model"]["name"]
            ).model_uri
            
            print(f"Model logged to: {model_path}")
            return model_path
    
    def deploy_model(self, model_path: str) -> None:
        """Deploy model to production."""
        # In a real implementation, this would:
        # 1. Create a deployment in your serving infrastructure
        # 2. Update routing rules
        # 3. Set up monitoring
        
        # For this example, we'll just save deployment metadata
        deployment_info = {
            "model_path": model_path,
            "deployment_time": datetime.now().isoformat(),
            "deployed_by": os.environ.get("USER", "unknown"),
            "environment": self.config["deployment"]["environment"],
            "api_endpoint": self.config["deployment"]["api_endpoint"]
        }
        
        deployment_path = f"deployment_{self.run_id}.yaml"
        with open(deployment_path, "w") as f:
            yaml.dump(deployment_info, f)
            
        print(f"Model deployed: {deployment_info}")
    
    def run_pipeline(self) -> None:
        """Run the complete ML pipeline."""
        # Extract data
        data = self.extract_data()
        
        # Validate data
        if not self.validate_data(data):
            print("Pipeline stopped due to data validation failures")
            return
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train, y_train)
        
        # Train model
        model = self.train_model(X_train, y_train, best_params)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Check if model meets performance threshold
        if metrics["f1"] < self.config["model"]["minimum_f1_score"]:
            print(f"Model performance below threshold: {metrics['f1']} < {self.config['model']['minimum_f1_score']}")
            return
        
        # Log model
        model_path = self.log_model(model, X_test, metrics)
        
        # Deploy model
        self.deploy_model(model_path)
        
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run_pipeline()
