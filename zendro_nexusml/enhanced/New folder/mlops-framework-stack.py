# mlflow_plus - A Custom MLOps Framework
# Core modules structure

# setup.py
import setuptools

setuptools.setup(
    name="mlflow_plus",
    version="0.1.0",
    author="ML Engineer",
    author_email="engineer@mlflowplus.com",
    description="A comprehensive MLOps framework",
    packages=setuptools.find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "mlflow>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "docker>=6.0.0",
        "kubernetes>=20.0.0",
        "fastapi>=0.90.0",
        "sqlalchemy>=2.0.0",
        "boto3>=1.20.0",
        "google-cloud-storage>=2.0.0",
        "azureml-core>=1.40.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mlflow-plus=mlflow_plus.cli:main",
        ],
    },
)

# mlflow_plus/config.py
import os
import yaml
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any


class StorageConfig(BaseModel):
    type: str = Field(..., description="Storage type: local, s3, gcs, azure")
    path: str = Field(..., description="Base path for storage")
    credentials: Optional[Dict[str, str]] = Field(None, description="Storage credentials")


class OrchestratorConfig(BaseModel):
    type: str = Field(..., description="Orchestrator type: local, kubernetes, airflow")
    config: Dict[str, Any] = Field(default_factory=dict, description="Orchestrator config")


class DatabaseConfig(BaseModel):
    type: str = Field(..., description="Database type: sqlite, mysql, postgresql")
    connection_string: str = Field(..., description="Database connection string")


class MLConfig(BaseModel):
    experiment_name: str = Field(..., description="ML experiment name")
    metrics: List[str] = Field(default_factory=list, description="Metrics to track")
    tags: Dict[str, str] = Field(default_factory=dict, description="Experiment tags")


class Config(BaseModel):
    storage: StorageConfig
    orchestrator: OrchestratorConfig
    database: DatabaseConfig
    ml: MLConfig
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")


def load_config(config_path: str) -> Config:
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Save config to YAML file"""
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)


# Sample config.yaml
sample_config = """
storage:
  type: s3
  path: s3://mlflow-plus/
  credentials:
    aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}

orchestrator:
  type: kubernetes
  config:
    namespace: mlflow-plus
    service_account: ml-service-account

database:
  type: postgresql
  connection_string: postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

ml:
  experiment_name: my-experiment
  metrics:
    - accuracy
    - precision
    - recall
    - f1
  tags:
    team: ml-team
    project: customer-segmentation

environment:
  PYTHONPATH: ${PYTHONPATH}:/app
  LOG_LEVEL: INFO
"""

# mlflow_plus/storage/__init__.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, BinaryIO, Optional
import os
import shutil
import mlflow


class Storage(ABC):
    """Base class for all storage implementations"""
    
    @abstractmethod
    def save_model(self, model_path: str, destination: str) -> str:
        """Save model to storage"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str, destination: str) -> str:
        """Load model from storage"""
        pass
    
    @abstractmethod
    def save_artifact(self, artifact_path: str, destination: str) -> str:
        """Save artifact to storage"""
        pass
    
    @abstractmethod
    def load_artifact(self, artifact_path: str, destination: str) -> str:
        """Load artifact from storage"""
        pass
    
    @abstractmethod
    def list_artifacts(self, path: str) -> List[str]:
        """List artifacts in storage"""
        pass


class LocalStorage(Storage):
    """Local file system storage"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model_path: str, destination: str) -> str:
        dest_path = os.path.join(self.base_path, destination)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(model_path, dest_path)
        return dest_path
    
    def load_model(self, model_path: str, destination: str) -> str:
        source_path = os.path.join(self.base_path, model_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source_path, destination)
        return destination
    
    def save_artifact(self, artifact_path: str, destination: str) -> str:
        dest_path = os.path.join(self.base_path, destination)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(artifact_path, dest_path)
        return dest_path
    
    def load_artifact(self, artifact_path: str, destination: str) -> str:
        source_path = os.path.join(self.base_path, artifact_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source_path, destination)
        return destination
    
    def list_artifacts(self, path: str) -> List[str]:
        full_path = os.path.join(self.base_path, path)
        if not os.path.exists(full_path):
            return []
        
        if os.path.isfile(full_path):
            return [os.path.basename(full_path)]
        
        result = []
        for root, _, files in os.walk(full_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), self.base_path)
                result.append(rel_path)
        return result


class S3Storage(Storage):
    """AWS S3 storage"""
    
    def __init__(self, base_path: str, credentials: Optional[Dict[str, str]] = None):
        import boto3
        self.base_path = base_path.rstrip('/')
        self.credentials = credentials or {}
        
        # Extract bucket and prefix from path
        if base_path.startswith('s3://'):
            parts = base_path[5:].split('/', 1)
            self.bucket = parts[0]
            self.prefix = parts[1] if len(parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 path: {base_path}")
        
        # Initialize S3 client
        self.s3 = boto3.client('s3', **self.credentials)
    
    def save_model(self, model_path: str, destination: str) -> str:
        s3_key = f"{self.prefix}/{destination}"
        self.s3.upload_file(model_path, self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"
    
    def load_model(self, model_path: str, destination: str) -> str:
        if model_path.startswith('s3://'):
            parts = model_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = self.bucket
            key = f"{self.prefix}/{model_path}"
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        self.s3.download_file(bucket, key, destination)
        return destination
    
    def save_artifact(self, artifact_path: str, destination: str) -> str:
        s3_key = f"{self.prefix}/{destination}"
        self.s3.upload_file(artifact_path, self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"
    
    def load_artifact(self, artifact_path: str, destination: str) -> str:
        if artifact_path.startswith('s3://'):
            parts = artifact_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = self.bucket
            key = f"{self.prefix}/{artifact_path}"
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        self.s3.download_file(bucket, key, destination)
        return destination
    
    def list_artifacts(self, path: str) -> List[str]:
        prefix = f"{self.prefix}/{path}"
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'][len(self.prefix)+1:] for obj in response['Contents']]


# Factory function to create storage instances
def get_storage(config: Any) -> Storage:
    """Create storage instance based on config"""
    if config.type == "local":
        return LocalStorage(config.path)
    elif config.type == "s3":
        return S3Storage(config.path, config.credentials)
    elif config.type == "gcs":
        from mlflow_plus.storage.gcs import GCSStorage
        return GCSStorage(config.path, config.credentials)
    elif config.type == "azure":
        from mlflow_plus.storage.azure import AzureStorage
        return AzureStorage(config.path, config.credentials)
    else:
        raise ValueError(f"Unsupported storage type: {config.type}")

# mlflow_plus/pipeline/__init__.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import inspect
import os
import tempfile
import json
import yaml
import pickle
import uuid
import datetime
import logging
import mlflow
from mlflow_plus.storage import Storage, get_storage
from mlflow_plus.config import Config, load_config


class Step(ABC):
    """Base class for all pipeline steps"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.inputs = {}
        self.outputs = {}
        self.logger = logging.getLogger(f"mlflow_plus.step.{name}")
    
    @abstractmethod
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the step with the given inputs"""
        pass
    
    def set_input(self, name: str, value: Any) -> None:
        """Set input for the step"""
        self.inputs[name] = value
    
    def get_output(self, name: str) -> Any:
        """Get output from the step"""
        if name not in self.outputs:
            raise ValueError(f"Output '{name}' not found in step '{self.name}'")
        return self.outputs[name]


class DataLoader(Step):
    """Data loading step"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.source = self.config.get("source", "file")
        self.path = self.config.get("path")
        self.format = self.config.get("format", "csv")
    
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load data from source"""
        import pandas as pd
        
        self.logger.info(f"Loading data from {self.source}: {self.path}")
        
        if self.source == "file":
            if self.format == "csv":
                data = pd.read_csv(self.path)
            elif self.format == "parquet":
                data = pd.read_parquet(self.path)
            elif self.format == "json":
                data = pd.read_json(self.path)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
        
        elif self.source == "database":
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.config["connection_string"])
            query = self.config.get("query", f"SELECT * FROM {self.config.get('table')}")
            data = pd.read_sql(query, engine)
        
        else:
            raise ValueError(f"Unsupported source: {self.source}")
        
        self.outputs["data"] = data
        return self.outputs


class DataPreprocessor(Step):
    """Data preprocessing step"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.operations = self.config.get("operations", [])
    
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Preprocess data"""
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
        
        inputs = inputs or self.inputs
        data = inputs["data"]
        
        self.logger.info(f"Preprocessing data with {len(self.operations)} operations")
        
        for op in self.operations:
            op_type = op["type"]
            
            if op_type == "drop_columns":
                columns = op["columns"]
                data = data.drop(columns=columns)
            
            elif op_type == "fill_na":
                columns = op.get("columns", data.columns)
                strategy = op.get("strategy", "mean")
                value = op.get("value")
                
                for col in columns:
                    if col not in data.columns:
                        continue
                    
                    if value is not None:
                        data[col] = data[col].fillna(value)
                    elif strategy == "mean":
                        data[col] = data[col].fillna(data[col].mean())
                    elif strategy == "median":
                        data[col] = data[col].fillna(data[col].median())
                    elif strategy == "mode":
                        data[col] = data[col].fillna(data[col].mode()[0])
            
            elif op_type == "scale":
                columns = op.get("columns", data.select_dtypes(include=np.number).columns)
                scaler_type = op.get("scaler", "standard")
                
                if scaler_type == "standard":
                    scaler = StandardScaler()
                elif scaler_type == "minmax":
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unsupported scaler: {scaler_type}")
                
                data[columns] = scaler.fit_transform(data[columns])
                self.outputs["scaler"] = scaler
            
            elif op_type == "encode":
                columns = op["columns"]
                encoder_type = op.get("encoder", "onehot")
                
                if encoder_type == "onehot":
                    encoder = OneHotEncoder(sparse_output=False)
                    encoded = encoder.fit_transform(data[columns])
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=encoder.get_feature_names_out(columns),
                        index=data.index
                    )
                    data = pd.concat([data.drop(columns=columns), encoded_df], axis=1)
                    self.outputs["encoder"] = encoder
                
                elif encoder_type == "label":
                    encoders = {}
                    for col in columns:
                        encoder = LabelEncoder()
                        data[col] = encoder.fit_transform(data[col])
                        encoders[col] = encoder
                    self.outputs["encoders"] = encoders
        
        self.outputs["data"] = data
        return self.outputs


class ModelTrainer(Step):
    """Model training step"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.model_type = self.config.get("model_type", "sklearn")
        self.algorithm = self.config.get("algorithm")
        self.parameters = self.config.get("parameters", {})
        self.target_column = self.config.get("target_column")
        self.feature_columns = self.config.get("feature_columns")
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
    
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train model"""
        from sklearn.model_selection import train_test_split
        
        inputs = inputs or self.inputs
        data = inputs["data"]
        
        # Determine features and target
        if self.target_column is None:
            raise ValueError("Target column must be specified")
        
        y = data[self.target_column]
        
        if self.feature_columns is None:
            X = data.drop(columns=[self.target_column])
        else:
            X = data[self.feature_columns]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        self.logger.info(f"Training {self.model_type}.{self.algorithm} model")
        
        # Train model
        if self.model_type == "sklearn":
            self._train_sklearn_model(X_train, y_train)
        elif self.model_type == "tensorflow":
            self._train_tensorflow_model(X_train, y_train)
        elif self.model_type == "pytorch":
            self._train_pytorch_model(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Evaluate model
        metrics = self._evaluate_model(X_test, y_test)
        
        self.outputs["model"] = self.model
        self.outputs["X_train"] = X_train
        self.outputs["X_test"] = X_test
        self.outputs["y_train"] = y_train
        self.outputs["y_test"] = y_test
        self.outputs["metrics"] = metrics
        self.outputs["feature_columns"] = X.columns.tolist()
        
        return self.outputs
    
    def _train_sklearn_model(self, X_train, y_train):
        """Train scikit-learn model"""
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        
        models = {
            "linear_regression": LinearRegression,
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestClassifier,
            "random_forest_regressor": RandomForestRegressor,
            "svc": SVC,
            "svr": SVR,
            "decision_tree_classifier": DecisionTreeClassifier,
            "decision_tree_regressor": DecisionTreeRegressor,
            "knn_classifier": KNeighborsClassifier,
            "knn_regressor": KNeighborsRegressor,
        }
        
        if self.algorithm not in models:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        model_class = models[self.algorithm]
        self.model = model_class(**self.parameters)
        self.model.fit(X_train, y_train)
    
    def _train_tensorflow_model(self, X_train, y_train):
        """Train TensorFlow model"""
        # Implement TensorFlow model training
        # This is a simplified example
        import tensorflow as tf
        
        # Convert to tensors
        X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
        
        # Build model
        input_dim = X_train.shape[1]
        layers = self.parameters.get("layers", [64, 32])
        activation = self.parameters.get("activation", "relu")
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(layers[0], input_dim=input_dim, activation=activation))
        
        for units in layers[1:]:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        
        if self.algorithm == "classifier":
            # Classification
            num_classes = len(y_train.unique())
            if num_classes == 2:
                model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
                loss = "binary_crossentropy"
            else:
                model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
                loss = "sparse_categorical_crossentropy"
        else:
            # Regression
            model.add(tf.keras.layers.Dense(1))
            loss = "mse"
        
        # Compile and train
        model.compile(
            optimizer=self.parameters.get("optimizer", "adam"),
            loss=loss,
            metrics=self.parameters.get("metrics", ["accuracy"])
        )
        
        model.fit(
            X_train_tensor, 
            y_train_tensor,
            epochs=self.parameters.get("epochs", 10),
            batch_size=self.parameters.get("batch_size", 32),
            verbose=1
        )
        
        self.model = model
    
    def _train_pytorch_model(self, X_train, y_train):
        """Train PyTorch model"""
        # Simplified PyTorch implementation
        # In a real system, this would be more robust
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        
        # Define model
        input_dim = X_train.shape[1]
        layers = self.parameters.get("layers", [64, 32])
        activation = self.parameters.get("activation", "relu")
        
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        
        if activation not in activations:
            activation = "relu"
        
        activation_fn = activations[activation]
        
        class SimpleNN(nn.Module):
            def __init__(self, input_dim, layers, activation_fn, output_dim=1):
                super(SimpleNN, self).__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                self.layers.append(nn.Linear(input_dim, layers[0]))
                
                # Hidden layers
                for i in range(len(layers) - 1):
                    self.layers.append(nn.Linear(layers[i], layers[i+1]))
                
                # Output layer
                self.layers.append(nn.Linear(layers[-1], output_dim))
                
                self.activation = activation_fn
            
            def forward(self, x):
                for i in range(len(self.layers) - 1):
                    x = self.activation(self.layers[i](x))
                
                # Last layer without activation for regression
                # or with specific activation for classification
                x = self.layers[-1](x)
                return x
        
        # Create model instance
        self.model = SimpleNN(input_dim, layers, activation_fn)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        epochs = self.parameters.get("epochs", 100)
        
        # Train model
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def _evaluate_model(self, X_test, y_test):
        """Evaluate model and return metrics"""
        import numpy as np
        
        if self.model_type == "sklearn":
            y_pred = self.model.predict(X_test)
            
            # Classification metrics
            if hasattr(self.model, "predict_proba"):
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                }
                
                # ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        y_prob = self.model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                    except:
                        pass
            
            # Regression metrics
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics = {
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                }
        
        elif self.model_type in ["tensorflow", "pytorch"]:
            # Simplified evaluation for deep learning models
            if self.model_type == "tensorflow":
                import tensorflow as tf
                X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
                y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)
                
                evaluation = self.model.evaluate(X_test_tensor, y_test_tensor)
                metrics = {
                    "loss": float(evaluation[0]),
                    "accuracy": float(evaluation[1]) if len(evaluation) > 1 else None
                }
            
            else:  # PyTorch
                import torch
                import torch.nn as nn
                
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
                    
                    criterion = nn.MSELoss()
                    self.model.eval()
                    outputs = self.model(X_test_tensor).squeeze()
                    loss = criterion(outputs, y_test_tensor)
                    
                    metrics = {
                        "mse": float(loss.item()),
                        "rmse": float(torch.sqrt(loss).item())
                    }
        
        return metrics


class ModelDeployer(Step):
    """Model deployment step"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.deployment_type = self.config.get("deployment_type", "mlflow")
        self.model_path = self.config.get("model_path", "models")
        self.version = self.config.get("version", "latest")
        self.serving_config = self.config.get("serving", {})
    
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy model"""
        import pickle
        import mlflow
        import tempfile
        import os
        
        inputs = inputs or self.inputs
        model = inputs["model"]
        metrics = inputs.get("metrics", {})
        feature_columns = inputs.get("feature_columns", [])
        
        self.logger.info(f"Deploying model with {self.deployment_type}")
        
        # Save model locally first
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = os.path.join(temp_dir, "model.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                "metrics": metrics,
                "feature_columns": feature_columns,
                "deployment_time": datetime.datetime.now().isoformat(),
                "deployment_type": self.deployment_type,
                "model_type": inputs.get("model_type", "unknown"),
                "version": self.version
            }
            
            metadata_file = os.path.join(temp_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
            
            # Deploy based on deployment type
            if self.deployment_type == "mlflow":
                self._deploy_mlflow(model, temp_dir, model_file, metadata)
            elif self.deployment_type == "fastapi":
                self._deploy_fastapi(model, temp_dir, model_file, metadata)
            elif self.deployment_type == "docker":
                self._deploy_docker(model, temp_dir, model_file, metadata)
            elif self.deployment_type == "kubernetes":
                self._deploy_kubernetes(model, temp_dir, model_file, metadata)
            else:
                raise ValueError(f"Unsupported deployment type: {self.deployment_type}")
        
        return self.outputs
    