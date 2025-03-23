# config.py - Framework Configuration
import os
from pathlib import Path

class MLOpsConfig:
    # Base paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
    
    # MLflow settings
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
    
    # Model serving
    MODEL_SERVING_PORT = int(os.environ.get("MODEL_SERVING_PORT", 8000))
    
    # Monitoring
    PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", 9090))
    GRAFANA_PORT = int(os.environ.get("GRAFANA_PORT", 3000))
    
    # Feature store
    FEATURE_STORE_TYPE = os.environ.get("FEATURE_STORE_TYPE", "local")  # local, feast, hopsworks
    
    # Data validation
    SCHEMA_DIR = os.path.join(BASE_DIR, "schemas")
    
    # CI/CD
    DOCKER_REGISTRY = os.environ.get("DOCKER_REGISTRY", "localhost:5000")
    
    @classmethod
    def make_dirs(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.EXPERIMENTS_DIR, cls.SCHEMA_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# data_pipeline.py - Data Ingestion and Processing
import pandas as pd
import great_expectations as ge
from datetime import datetime
import hashlib
import json
import mlflow
from config import MLOpsConfig

class DataPipeline:
    def __init__(self, config=None):
        self.config = config or MLOpsConfig()
        self.config.make_dirs()
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def ingest_data(self, source_path, target_name=None):
        """Ingest data from source and store it with versioning"""
        if source_path.endswith('.csv'):
            df = pd.read_csv(source_path)
        elif source_path.endswith('.parquet'):
            df = pd.read_parquet(source_path)
        else:
            raise ValueError(f"Unsupported file format: {source_path}")
        
        # Generate version hash based on data content
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        target_name = target_name or Path(source_path).stem
        
        # Save with version information
        versioned_path = os.path.join(
            self.config.DATA_DIR, 
            f"{target_name}_{self.version}_{data_hash}.parquet"
        )
        df.to_parquet(versioned_path)
        
        # Track data version in MLflow
        with mlflow.start_run(run_name=f"data_ingestion_{target_name}"):
            mlflow.log_param("data_source", source_path)
            mlflow.log_param("data_version", self.version)
            mlflow.log_param("data_hash", data_hash)
            mlflow.log_artifact(versioned_path)
        
        return versioned_path, df
    
    def validate_data(self, df, schema_name):
        """Validate data against schema"""
        schema_path = os.path.join(self.config.SCHEMA_DIR, f"{schema_name}.json")
        
        # Create schema if it doesn't exist
        if not os.path.exists(schema_path):
            schema = self._infer_schema(df)
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
        
        # Load schema and validate
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        ge_df = ge.from_pandas(df)
        validation_result = self._validate_against_schema(ge_df, schema)
        
        return validation_result
    
    def _infer_schema(self, df):
        """Infer schema from DataFrame"""
        schema = {
            "columns": {},
            "row_count_range": [df.shape[0] * 0.9, df.shape[0] * 1.1]  # Allow 10% variation
        }
        
        for col in df.columns:
            col_schema = {
                "type": str(df[col].dtype),
                "nullable": df[col].isnull().any()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_schema.update({
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                })
            
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                col_schema["unique_values"] = df[col].dropna().unique().tolist()
            
            schema["columns"][col] = col_schema
            
        return schema
    
    def _validate_against_schema(self, ge_df, schema):
        """Validate DataFrame using Great Expectations and schema"""
        results = {}
        
        # Check row count
        row_count = ge_df.shape[0]
        results["row_count"] = {
            "valid": schema["row_count_range"][0] <= row_count <= schema["row_count_range"][1],
            "actual": row_count,
            "expected": schema["row_count_range"]
        }
        
        # Check columns
        expected_columns = set(schema["columns"].keys())
        actual_columns = set(ge_df.columns)
        results["columns_match"] = {
            "valid": expected_columns == actual_columns,
            "missing": list(expected_columns - actual_columns),
            "extra": list(actual_columns - expected_columns)
        }
        
        # Check column properties
        results["column_validation"] = {}
        for col, col_schema in schema["columns"].items():
            if col not in ge_df.columns:
                continue
                
            col_results = {}
            
            # Check type
            actual_type = str(ge_df[col].dtype)
            type_valid = actual_type == col_schema["type"]
            col_results["type"] = {
                "valid": type_valid,
                "actual": actual_type,
                "expected": col_schema["type"]
            }
            
            # Check numeric properties
            if pd.api.types.is_numeric_dtype(ge_df[col]) and "min" in col_schema:
                # Min/max validation
                min_valid = ge_df[col].min() >= col_schema["min"] * 0.9  # Allow 10% tolerance
                max_valid = ge_df[col].max() <= col_schema["max"] * 1.1  # Allow 10% tolerance
                
                col_results["range"] = {
                    "valid": min_valid and max_valid,
                    "actual": [float(ge_df[col].min()), float(ge_df[col].max())],
                    "expected": [col_schema["min"], col_schema["max"]]
                }
            
            results["column_validation"][col] = col_results
            
        return results

# model_training.py - Training Pipeline
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from config import MLOpsConfig

class ModelTraining:
    def __init__(self, config=None):
        self.config = config or MLOpsConfig()
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
    
    def train(self, X, y, model, model_name, params=None, metrics=None, test_size=0.2):
        """Train a model with MLflow tracking"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            run_id = run.info.run_id
            
            # Log parameters
            params = params or {}
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Log metrics
            y_pred = model.predict(X_test)
            metrics_dict = self._calculate_metrics(y_test, y_pred, metrics)
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature names
            if hasattr(X, 'columns'):
                mlflow.log_param("features", list(X.columns))
            
            print(f"Model trained and logged with run_id: {run_id}")
            return run_id, metrics_dict
    
    def _calculate_metrics(self, y_true, y_pred, metrics=None):
        """Calculate specified metrics"""
        from sklearn import metrics as sk_metrics
        
        results = {}
        metrics = metrics or ["accuracy", "precision", "recall", "f1"]
        
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = sk_metrics.accuracy_score(y_true, y_pred)
            elif metric == "precision":
                results[metric] = sk_metrics.precision_score(y_true, y_pred, average='weighted')
            elif metric == "recall":
                results[metric] = sk_metrics.recall_score(y_true, y_pred, average='weighted')
            elif metric == "f1":
                results[metric] = sk_metrics.f1_score(y_true, y_pred, average='weighted')
            elif metric == "rmse":
                results[metric] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))
            elif metric == "mae":
                results[metric] = sk_metrics.mean_absolute_error(y_true, y_pred)
            
        return results
    
    def register_model(self, run_id, model_name, stage="Staging"):
        """Register a model from a run to the model registry"""
        result = mlflow.register_model(
            f"runs:/{run_id}/model",
            model_name
        )
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage
        )
        
        print(f"Model {model_name} version {result.version} registered as {stage}")
        return result.version

# model_serving.py - Model Deployment and Serving
import mlflow
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
from config import MLOpsConfig

class ModelServer:
    def __init__(self, model_name, model_version="latest", config=None):
        self.config = config or MLOpsConfig()
        self.model_name = model_name
        self.model_version = model_version
        self.app = FastAPI(title=f"{model_name} API", version="1.0.0")
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        
        # Load the model
        self.load_model()
        
        # Register API endpoints
        self._register_endpoints()
    
    def load_model(self):
        """Load model from MLflow model registry"""
        client = mlflow.tracking.MlflowClient()
        
        if self.model_version == "latest":
            # Get latest model version
            models = client.search_model_versions(f"name='{self.model_name}'")
            production_models = [m for m in models if m.current_stage == "Production"]
            
            if production_models:
                self.model_version = production_models[0].version
            else:
                # Get latest version
                versions = [int(m.version) for m in models]
                self.model_version = str(max(versions)) if versions else "1"
        
        # Load the model
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model metadata
            model_info = client.get_model_version(self.model_name, self.model_version)
            run_id = model_info.run_id
            run = client.get_run(run_id)
            
            # Extract feature names if available
            self.feature_names = run.data.params.get("features", "").strip("[]").replace("'", "").split(", ")
            
            print(f"Loaded {self.model_name} version {self.model_version}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _register_endpoints(self):
        """Register API endpoints"""
        # Create dynamic Pydantic model based on feature names
        features = {name: (float, ...) for name in self.feature_names} if hasattr(self, 'feature_names') else {"features": (List[float], ...)}
        PredictionRequest = create_model("PredictionRequest", **features)
        
        class PredictionResponse(BaseModel):
            prediction: Any
            prediction_time: str
            model_version: str
        
        @self.app.get("/")
        def root():
            return {
                "message": f"{self.model_name} API is running",
                "model_version": self.model_version,
                "endpoints": ["/predict", "/health"]
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        def predict(request: PredictionRequest):
            try:
                # Convert input to DataFrame
                if hasattr(self, 'feature_names'):
                    input_data = pd.DataFrame([request.dict()])
                else:
                    input_data = pd.DataFrame([request.features]).T
                
                # Make prediction
                prediction = self.model.predict(input_data)
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.tolist()
                
                # Log prediction for monitoring
                self._log_prediction(request.dict(), prediction)
                
                return {
                    "prediction": prediction[0] if isinstance(prediction, list) else prediction,
                    "prediction_time": datetime.now().isoformat(),
                    "model_version": self.model_version
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        def health():
            return {"status": "healthy", "model": self.model_name, "version": self.model_version}
    
    def _log_prediction(self, input_data, prediction):
        """Log prediction for monitoring"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input": input_data,
            "prediction": prediction
        }
        
        # In a real implementation, this would send to a logging system
        # For example: Kafka, Elasticsearch, or a database
        print(f"Prediction log: {json.dumps(log_data)}")
    
    def start(self, host="0.0.0.0", port=None):
        """Start the model server"""
        port = port or self.config.MODEL_SERVING_PORT
        uvicorn.run(self.app, host=host, port=port)

# monitoring.py - Model Monitoring
import pandas as pd
import numpy as np
from scipy import stats
import json
import time
from datetime import datetime
import threading
from config import MLOpsConfig

class ModelMonitor:
    def __init__(self, model_name, model_version, config=None):
        self.config = config or MLOpsConfig()
        self.model_name = model_name
        self.model_version = model_version
        
        # Initialize storage for prediction data
        self.predictions = []
        self.baseline_data = None
        self.drift_thresholds = {}
        self.alerts = []
        
        # Set up monitoring thread
        self.monitoring_active = False
    
    def load_baseline(self, baseline_path):
        """Load baseline data for drift detection"""
        self.baseline_data = pd.read_parquet(baseline_path)
        
        # Calculate statistical properties for each feature
        for column in self.baseline_data.columns:
            if pd.api.types.is_numeric_dtype(self.baseline_data[column]):
                self.drift_thresholds[column] = {
                    "mean": self.baseline_data[column].mean(),
                    "std": self.baseline_data[column].std(),
                    "p05": self.baseline_data[column].quantile(0.05),
                    "p95": self.baseline_data[column].quantile(0.95),
                    "threshold": 0.05  # p-value threshold for KS test
                }
    
    def add_prediction(self, input_data, prediction, timestamp=None):
        """Add a prediction for monitoring"""
        timestamp = timestamp or datetime.now().isoformat()
        
        record = {
            "timestamp": timestamp,
            "input": input_data,
            "prediction": prediction
        }
        
        self.predictions.append(record)
        
        # Check for drift if we have enough data
        if len(self.predictions) % 100 == 0:
            self.check_drift()
    
    def check_drift(self):
        """Check for data and concept drift"""
        if not self.baseline_data is not None or len(self.predictions) < 50:
            return
        
        # Extract input data from predictions
        recent_inputs = []
        for record in self.predictions[-100:]:
            if isinstance(record["input"], dict):
                recent_inputs.append(record["input"])
        
        if not recent_inputs:
            return
            
        # Convert to DataFrame
        recent_df = pd.DataFrame(recent_inputs)
        
        # Check each numeric feature for drift
        drift_detected = False
        drift_report = {"timestamp": datetime.now().isoformat(), "features": {}}
        
        for column in self.drift_thresholds:
            if column in recent_df.columns and pd.api.types.is_numeric_dtype(recent_df[column]):
                # Get baseline and recent data
                baseline_values = self.baseline_data[column].dropna()
                recent_values = recent_df[column].dropna()
                
                if len(recent_values) < 10:
                    continue
                    
                # Basic statistics comparison
                recent_mean = recent_values.mean()
                recent_std = recent_values.std()
                baseline_mean = self.drift_thresholds[column]["mean"]
                baseline_std = self.drift_thresholds[column]["std"]
                
                # Calculate z-score for mean shift
                mean_z_score = abs(recent_mean - baseline_mean) / baseline_std
                
                # Perform KS test
                ks_statistic, p_value = stats.ks_2samp(baseline_values, recent_values)
                
                # Check if drift is detected
                threshold = self.drift_thresholds[column]["threshold"]
                if p_value < threshold or mean_z_score > 3:
                    drift_detected = True
                    drift_report["features"][column] = {
                        "drift_detected": True,
                        "p_value": p_value,
                        "ks_statistic": ks_statistic,
                        "mean_shift": {
                            "baseline": baseline_mean,
                            "recent": recent_mean,
                            "z_score": mean_z_score
                        }
                    }
        
        if drift_detected:
            drift_report["status"] = "DRIFT_DETECTED"
            self.alerts.append(drift_report)
            self._send_alert(drift_report)
        
        return drift_report
    
    def _send_alert(self, alert_data):
        """Send an alert (would be implemented with a notification system)"""
        print(f"⚠️ ALERT: Data drift detected for model {self.model_name} v{self.model_version}")
        print(json.dumps(alert_data, indent=2))
        
        # In a real implementation, this would send to an alerting system
        # For example: email, Slack, PagerDuty, etc.
    
    def start_monitoring(self, interval_seconds=300):
        """Start background monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.check_drift()
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        print(f"Started monitoring for model {self.model_name} v{self.model_version}")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        print(f"Stopped monitoring for model {self.model_name} v{self.model_version}")

# mlops_manager.py - Unified Manager for MLOps Framework
from data_pipeline import DataPipeline
from model_training import ModelTraining
from model_serving import ModelServer
from monitoring import ModelMonitor
from config import MLOpsConfig
import os
import mlflow
import subprocess
from pathlib import Path

class MLOpsManager:
    def __init__(self, project_name, config=None):
        self.project_name = project_name
        self.config = config or MLOpsConfig()
        self.config.make_dirs()
        
        # Initialize components
        self.data_pipeline = DataPipeline(self.config)
        self.model_training = ModelTraining(self.config)
        
        # Set MLflow experiment
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(project_name)
        
        self.model_server = None
        self.model_monitor = None
    
    def setup_mlflow(self):
        """Set up MLflow server if it's not running"""
        try:
            # Check if MLflow server is running
            import requests
            requests.get(self.config.MLFLOW_TRACKING_URI)
            print(f"MLflow already running at {self.config.MLFLOW_TRACKING_URI}")
        except:
            # Start MLflow server
            mlflow_dir = os.path.join(self.config.BASE_DIR, "mlflow")
            Path(mlflow_dir).mkdir(parents=True, exist_ok=True)
            
            cmd = f"mlflow server --backend-store-uri sqlite:///{mlflow_dir}/mlflow.db --default-artifact-root {mlflow_dir}/artifacts --host 0.0.0.0 --port 5000"
            
            # Start as subprocess
            print(f"Starting MLflow server: {cmd}")
            subprocess.Popen(cmd, shell=True)
            print("Waiting for MLflow server to start...")
            import time
            time.sleep(5)
    
    def ingest_and_validate_data(self, source_path, target_name=None, schema_name=None):
        """Ingest and validate data"""
        # Ingest data
        versioned_path, df = self.data_pipeline.ingest_data(source_path, target_name)
        
        # Validate data
        schema_name = schema_name or (target_name or Path(source_path).stem)
        validation_result = self.data_pipeline.validate_data(df, schema_name)
        
        # Log validation results
        with mlflow.start_run(run_name=f"data_validation_{schema_name}"):
            mlflow.log_param("data_path", versioned_path)
            mlflow.log_param("schema_name", schema_name)
            mlflow.log_dict(validation_result, "validation_result.json")
            
            # Check if validation passed
            columns_valid = validation_result["columns_match"]["valid"]
            row_count_valid = validation_result["row_count"]["valid"]
            mlflow.log_param("validation_passed", columns_valid and row_count_valid)
        
        return versioned_path, df, validation_result
    
    def train_and_register_model(self, X, y, model, model_name, params=None, metrics=None):
        """Train and register a model"""
        # Train model
        run_id, metrics_dict = self.model_training.train(X, y, model, model_name, params, metrics)
        
        # Register model
        version = self.model_training.register_model(run_id, model_name)
        
        return run_id, version, metrics_dict
    
    def deploy_model(self, model_name, model_version="latest", port=None):
        """Deploy a model for serving"""
        # Initialize model server
        self.model_server = ModelServer(model_name, model_version, self.config)
        
        # Initialize model monitor
        self.model_monitor = ModelMonitor(model_name, model_version, self.config)
        
        # Start monitoring
        self.model_monitor.start_monitoring()
        
        # Start server (this will block until interrupted)
        port = port or self.config.MODEL_SERVING_PORT
        self.model_server.start(port=port)
    
    def create_deployment_artifacts(self, model_name, model_version="latest", output_dir=None):
        """Create deployment artifacts (Dockerfile, docker-compose.yml)"""
        output_dir = output_dir or os.path.join(self.config.BASE_DIR, "deployment")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create Dockerfile
        dockerfile_content = f"""FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install mlflow fastapi uvicorn pandas scikit-learn numpy scipy

# Set environment variables
ENV MODEL_NAME={model_name}
ENV MODEL_VERSION={model_version}
ENV MLFLOW_TRACKING_URI={self.config.MLFLOW_TRACKING_URI}

# Copy application code
COPY model_serving.py /app/
COPY config.py /app/

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-c", "from model_serving import ModelServer; ModelServer(model_name='$MODEL_NAME', model_version='$MODEL_VERSION').start()"]
"""
        
        # Create docker-compose.yml
        compose_content = f"""version: '3'

services:
  mlflow:
    image: python:3.9-slim
    command: sh -c "pip install mlflow && mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow
    networks:
      - mlops-network

  model-service:
    build: .