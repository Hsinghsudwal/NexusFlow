import os
import logging
import yaml
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")


class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config):
        self.config = config
        self.base_path = self.config.get("folder_path", {}).get(
            "artifacts", "artifacts"
        )
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")

    def save_artifact(
        self,
        artifact: Any,
        subdir: str,
        name: str,
    ) -> None:
        """Save an artifact in the specified format."""
        artifact_dir = os.path.join(self.base_path, subdir)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        if name.endswith(".pkl"):
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
        elif name.endswith(".csv"):
            if isinstance(artifact, pd.DataFrame):
                artifact.to_csv(artifact_path, index=False)
            else:
                raise ValueError("CSV format only supports pandas DataFrames.")
        else:
            raise ValueError(f"Unsupported format for {name}")
        logging.info(f"Artifact '{name}' saved to {artifact_path}")

    def load_artifact(
        self,
        subdir: str,
        name: str,
    ):
        """Load an artifact in the specified format."""
        artifact_path = os.path.join(self.base_path, subdir, name)
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
            else:
                raise ValueError(f"Unsupported format for {name}")
            logging.info(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
        else:
            logging.warning(f"Artifact '{name}' not found in {artifact_path}")
            return None


class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)

    def data_ingestion(self, path):
        # Define paths for artifacts
        raw_path = self.config.get("folder_path", {}).get("raw_data", "raw_data")
        raw_train_filename = self.config.get("filenames", {}).get("raw_train", "train_data.csv")
        raw_test_filename = self.config.get("filenames", {}).get("raw_test", "test_data.csv")
        
        # Load raw data
        df = pd.read_csv(path)
        # Split data
        test_size = self.config.get("base", {}).get("test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(
            f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )
        # Save raw artifacts
        self.artifact_store.save_artifact(
            train_data, subdir=raw_path, name=raw_train_filename 
        )
        self.artifact_store.save_artifact(
            test_data, subdir=raw_path, name=raw_test_filename
        )
        logging.info("Data ingestion completed")
        return train_data, test_data


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        strategy = self.config.get("data_transformation", {}).get("missing_values_strategy", "mean")
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                if strategy == "mean":
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == "median":
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy == "zero":
                    df[column].fillna(0, inplace=True)
            else:
                # For categorical columns
                df[column].fillna(df[column].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        encoding_method = self.config.get("data_transformation", {}).get("encoding_method", "one_hot")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if encoding_method == "one_hot":
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif encoding_method == "label":
            for col in categorical_cols:
                df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Scale numerical features."""
        scaling_method = self.config.get("data_transformation", {}).get("scaling_method", "standard")
        
        if scaling_method == "standard":
            scaler = StandardScaler()
            
            # Get feature columns (exclude target column)
            target_col = self.config.get("data_transformation", {}).get("target_column", "target")
            feature_cols = [col for col in train_df.columns if col != target_col]
            
            # Scale features
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            
            return train_df, test_df, scaler
        
        return train_df, test_df, None
    
    def transform_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the data by handling missing values, encoding categorical features, and scaling."""
        # Define paths for artifacts
        processed_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = self.config.get("filenames", {}).get("processed_train", "processed_train.csv")
        processed_test_filename = self.config.get("filenames", {}).get("processed_test", "processed_test.csv")
        scaler_filename = self.config.get("filenames", {}).get("scaler", "scaler.pkl")
        
        logging.info("Starting data transformation...")
        
        # Handle missing values
        train_data = self._handle_missing_values(train_data)
        test_data = self._handle_missing_values(test_data)
        
        # Encode categorical features
        train_data = self._encode_categorical_features(train_data)
        test_data = self._encode_categorical_features(test_data)
        
        # Make sure test_data has the same columns as train_data
        for col in train_data.columns:
            if col not in test_data.columns:
                test_data[col] = 0
        
        # Make sure columns are in the same order
        test_data = test_data[train_data.columns]
        
        # Scale features
        train_data, test_data, scaler = self._scale_features(train_data, test_data)
        
        # Save artifacts
        self.artifact_store.save_artifact(
            train_data, subdir=processed_path, name=processed_train_filename
        )
        self.artifact_store.save_artifact(
            test_data, subdir=processed_path, name=processed_test_filename
        )
        
        if scaler:
            self.artifact_store.save_artifact(
                scaler, subdir=processed_path, name=scaler_filename
            )
        
        logging.info("Data transformation completed")
        return train_data, test_data


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)
    
    def train_model(self, train_data: pd.DataFrame) -> Any:
        """Train a machine learning model."""
        # Define paths for artifacts
        models_path = self.config.get("folder_path", {}).get("models", "models")
        model_filename = self.config.get("filenames", {}).get("model", "model.pkl")
        
        # Get model parameters
        model_type = self.config.get("model_training", {}).get("model_type", "random_forest")
        target_col = self.config.get("data_transformation", {}).get("target_column", "target")
        
        # Split features and target
        X = train_data.drop(target_col, axis=1)
        y = train_data[target_col]
        
        # Train model
        if model_type == "logistic_regression":
            model = LogisticRegression(
                C=self.config.get("model_training", {}).get("C", 1.0),
                max_iter=self.config.get("model_training", {}).get("max_iter", 100),
                random_state=42
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.config.get("model_training", {}).get("n_estimators", 100),
                max_depth=self.config.get("model_training", {}).get("max_depth", None),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logging.info(f"Training {model_type} model...")
        model.fit(X, y)
        
        # Save the model
        self.artifact_store.save_artifact(
            model, subdir=models_path, name=model_filename
        )
        
        logging.info("Model training completed")
        return model


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)
    
    def evaluate_model(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate a trained model."""
        # Define paths for artifacts
        metrics_path = self.config.get("folder_path", {}).get("metrics", "metrics")
        metrics_filename = self.config.get("filenames", {}).get("metrics", "metrics.json")
        
        # Get target column
        target_col = self.config.get("data_transformation", {}).get("target_column", "target")
        
        # Split features and target
        X_test = test_data.drop(target_col, axis=1)
        y_test = test_data[target_col]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Save metrics
        os.makedirs(os.path.join(self.artifact_store.base_path, metrics_path), exist_ok=True)
        metrics_file_path = os.path.join(self.artifact_store.base_path, metrics_path, metrics_filename)
        
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics


class Pipeline:
    def __init__(self, config_path: str):
        """Initialize the pipeline with a configuration file."""
        self.config = Config.load_file(config_path)
        self.data_ingestion = DataIngestion(self.config)
        self.data_transformation = DataTransformation(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
    
    def run(self, data_path: str) -> Dict[str, float]:
        """Run the full pipeline."""
        logging.info("Starting pipeline execution...")
        
        # Data ingestion
        train_data, test_data = self.data_ingestion.data_ingestion(data_path)
        
        # Data transformation
        processed_train_data, processed_test_data = self.data_transformation.transform_data(train_data, test_data)
        
        # Model training
        model = self.model_trainer.train_model(processed_train_data)
        
        # Model evaluation
        metrics = self.model_evaluator.evaluate_model(model, processed_test_data)
        
        logging.info("Pipeline execution completed")
        return metrics


class PredictionService:
    def __init__(self, config_path: str):
        """Initialize the prediction service with a configuration file."""
        self.config = Config.load_file(config_path)
        self.artifact_store = ArtifactStore(self.config)
    
    def load_model(self) -> Any:
        """Load the trained model."""
        models_path = self.config.get("folder_path", {}).get("models", "models")
        model_filename = self.config.get("filenames", {}).get("model", "model.pkl")
        
        model = self.artifact_store.load_artifact(subdir=models_path, name=model_filename)
        
        if model is None:
            raise ValueError("Model not found. Please train the model first.")
        
        return model
    
    def load_scaler(self) -> Any:
        """Load the feature scaler."""
        processed_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        scaler_filename = self.config.get("filenames", {}).get("scaler", "scaler.pkl")
        
        scaler = self.artifact_store.load_artifact(subdir=processed_path, name=scaler_filename)
        
        return scaler
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        # Load a sample of training data to get the structure
        processed_path = self.config.get("folder_path", {}).get("processed_data", "processed_data")
        processed_train_filename = self.config.get("filenames", {}).get("processed_train", "processed_train.csv")
        
        sample_train_data = self.artifact_store.load_artifact(
            subdir=processed_path, name=processed_train_filename
        )
        
        if sample_train_data is None:
            raise ValueError("Processed training data not found. Cannot determine feature structure.")
        
        # Get feature columns (exclude target column)
        target_col = self.config.get("data_transformation", {}).get("target_column", "target")
        feature_cols = [col for col in sample_train_data.columns if col != target_col]
        
        # Transform the input data similar to training data transformation
        transformer = DataTransformation(self.config)
        
        # Handle missing values
        input_data = transformer._handle_missing_values(input_data)
        
        # Encode categorical features
        input_data = transformer._encode_categorical_features(input_data)
        
        # Ensure input has all expected columns
        for col in feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Only keep necessary columns
        input_data = input_data[feature_cols]
        
        # Scale features if a scaler exists
        scaler = self.load_scaler()
        if scaler:
            input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
        
        return input_data
    
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        # Load model
        model = self.load_model()
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        predictions = model.predict(processed_input)
        
        return predictions


# Example configuration file (sample_config.yaml)
def create_sample_config():
    config = {
        "base": {
            "test_size": 0.2
        },
        "folder_path": {
            "artifacts": "artifacts",
            "raw_data": "raw_data",
            "processed_data": "processed_data",
            "models": "models",
            "metrics": "metrics"
        },
        "filenames": {
            "raw_train": "raw_train.csv",
            "raw_test": "raw_test.csv",
            "processed_train": "processed_train.csv",
            "processed_test": "processed_test.csv",
            "scaler": "scaler.pkl",
            "model": "model.pkl",
            "metrics": "metrics.json"
        },
        "data_transformation": {
            "target_column": "target",
            "missing_values_strategy": "mean",
            "encoding_method": "one_hot",
            "scaling_method": "standard"
        },
        "model_training": {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 10
        }
    }
    
    # Write config to file
    os.makedirs("config", exist_ok=True)
    with open("config/sample_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info("Sample configuration file created at 'config/sample_config.yaml'")
    return config


# Example usage
if __name__ == "__main__":
    # Create sample config if it doesn't exist
    if not os.path.exists("config/sample_config.yaml"):
        create_sample_config()
    
    # Run the pipeline
    pipeline = Pipeline("config/sample_config.yaml")
    
    # Example: you would replace this with your actual data path
    data_path = "data/sample_data.csv"
    
    try:
        metrics = pipeline.run(data_path)
        print(f"Pipeline execution completed successfully. Metrics: {metrics}")
        
        # Example of using the prediction service
        prediction_service = PredictionService("config/sample_config.yaml")
        
        # Example: load some new data for prediction
        # new_data = pd.read_csv("data/new_data.csv")
        # predictions = prediction_service.predict(new_data)
        # print(f"Predictions: {predictions}")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise
