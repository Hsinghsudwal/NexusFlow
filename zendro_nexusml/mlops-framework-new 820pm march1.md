# MLOps Framework: Building Modular, Reproducible ML Pipelines with Prefect

This framework provides an end-to-end approach for managing machine learning workflows from data ingestion to model monitoring, with a focus on modularity, reproducibility, and collaboration.

## 1. Framework Architecture

```
mlops_framework/
├── config/                      # Configuration files
│   ├── logging_config.yaml      # Logging configuration
│   └── pipeline_config.yaml     # Pipeline parameters
├── data/                        # Data storage
│   ├── raw/                     # Raw input data
│   ├── processed/               # Processed datasets
│   └── features/                # Extracted features
├── models/                      # Model storage
│   ├── trained/                 # Trained model artifacts
│   └── deployed/                # Models ready for deployment
├── pipelines/                   # Reusable pipeline components
│   ├── data_ingestion/          # Data collection processes
│   ├── data_validation/         # Data quality checks
│   ├── feature_engineering/     # Feature transformation
│   ├── model_training/          # Model training processes
│   ├── model_evaluation/        # Model performance assessment
│   └── model_deployment/        # Deployment workflows
├── monitoring/                  # Monitoring components
│   ├── data_drift/              # Data drift detection
│   ├── model_performance/       # Model performance tracking
│   └── system_metrics/          # System health metrics
├── utils/                       # Utility functions
│   ├── logging_utils.py         # Logging utilities
│   ├── metrics_utils.py         # Metrics calculation helpers
│   └── visualization_utils.py   # Visualization helpers
├── artifacts/                   # Pipeline artifacts
│   ├── metadata/                # Metadata storage
│   └── logs/                    # Execution logs
├── tests/                       # Testing suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── docs/                        # Documentation
├── app/                         # Model serving applications
├── deployment/                  # Deployment configuration
│   ├── docker/                  # Docker files
│   └── kubernetes/              # K8s configuration
├── notebooks/                   # Exploratory notebooks
├── prefect_flows/               # Prefect flow definitions
├── pyproject.toml               # Project dependencies
├── setup.py                     # Package installation script
└── README.md                    # Project documentation
```

## 2. Core Components

### 2.1. Data Management

Data management covers ingestion, versioning, validation, and preprocessing:

```python
# pipelines/data_ingestion/base_ingestion.py
import pandas as pd
from prefect import task, flow
import dvc.api

class DataIngestionBase:
    """Base class for data ingestion tasks."""
    
    def __init__(self, config):
        self.config = config
        
    @task
    def extract_data(self):
        """Extract data from source."""
        raise NotImplementedError
        
    @task
    def validate_data(self, data):
        """Validate data quality."""
        raise NotImplementedError
        
    @task
    def save_data(self, data, path):
        """Save data with versioning."""
        # Save the data
        data.to_parquet(path)
        
        # Version with DVC
        import subprocess
        subprocess.run(["dvc", "add", path])
        subprocess.run(["dvc", "push"])
        
        return path

@flow(name="Data Ingestion Flow")
def data_ingestion_flow(config):
    """Orchestrate the data ingestion process."""
    ingestion = DataIngestionBase(config)
    data = ingestion.extract_data()
    validated_data = ingestion.validate_data(data)
    path = ingestion.save_data(validated_data, config["output_path"])
    return path
```

### 2.2. Feature Engineering

Reusable feature transformations with versioning:

```python
# pipelines/feature_engineering/feature_store.py
import mlflow
from prefect import task, flow
import pandas as pd

class FeatureStore:
    """Manage feature creation, storage, and retrieval."""
    
    def __init__(self, config):
        self.config = config
        
    @task
    def create_features(self, data):
        """Apply feature transformations."""
        raise NotImplementedError
        
    @task
    def log_features(self, features, feature_group, version):
        """Log features to a registry."""
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("feature_group", feature_group)
            mlflow.log_param("version", version)
            # Log feature metadata
            mlflow.log_dict(features.describe().to_dict(), f"{feature_group}_stats.json")
            # Save feature data
            features.to_parquet(f"features/{feature_group}/v{version}/features.parquet")
            mlflow.log_artifact(f"features/{feature_group}/v{version}/features.parquet")
        
        return f"features/{feature_group}/v{version}/features.parquet"
        
    @task
    def retrieve_features(self, feature_group, version):
        """Retrieve features by group and version."""
        return pd.read_parquet(f"features/{feature_group}/v{version}/features.parquet")

@flow(name="Feature Engineering Flow")
def feature_engineering_flow(data, config):
    """Orchestrate the feature engineering process."""
    feature_store = FeatureStore(config)
    features = feature_store.create_features(data)
    path = feature_store.log_features(
        features, 
        config["feature_group"], 
        config["version"]
    )
    return path
```

### 2.3. Model Training

Training pipeline with experiment tracking:

```python
# pipelines/model_training/model_trainer.py
import mlflow
from prefect import task, flow
from sklearn.model_selection import train_test_split

class ModelTrainer:
    """Base class for model training."""
    
    def __init__(self, config):
        self.config = config
        mlflow.set_experiment(config["experiment_name"])
        
    @task
    def prepare_training_data(self, features_path):
        """Prepare data for training."""
        import pandas as pd
        features = pd.read_parquet(features_path)
        X = features.drop(self.config["target_column"], axis=1)
        y = features[self.config["target_column"]]
        return train_test_split(X, y, test_size=self.config["test_size"], random_state=42)
        
    @task
    def train_model(self, X_train, y_train):
        """Train the model."""
        raise NotImplementedError
        
    @task
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        raise NotImplementedError
        
    @task
    def log_model(self, model, metrics, run_name):
        """Log model to registry."""
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            for param, value in self.config.items():
                mlflow.log_param(param, value)
                
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Register model in registry
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                self.config["model_name"]
            )
        
        return mlflow.active_run().info.run_id

@flow(name="Model Training Flow")
def model_training_flow(features_path, config):
    """Orchestrate the model training process."""
    trainer = ModelTrainer(config)
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(features_path)
    model = trainer.train_model(X_train, y_train)
    metrics = trainer.evaluate_model(model, X_test, y_test)
    run_id = trainer.log_model(model, metrics, f"{config['model_name']}_training")
    return run_id
```

### 2.4. Model Evaluation and Validation

```python
# pipelines/model_evaluation/model_validator.py
import mlflow
from prefect import task, flow
import pandas as pd
import numpy as np

class ModelValidator:
    """Model validation and quality gates."""
    
    def __init__(self, config):
        self.config = config
        
    @task
    def load_model(self, run_id):
        """Load model from registry."""
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
    @task
    def load_test_data(self, features_path):
        """Load test data."""
        features = pd.read_parquet(features_path)
        X = features.drop(self.config["target_column"], axis=1)
        y = features[self.config["target_column"]]
        return X, y
        
    @task
    def compute_metrics(self, model, X, y):
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted'),
            "f1": f1_score(y, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_prob)
            
        return metrics
        
    @task
    def validate_metrics(self, metrics):
        """Apply quality gates to metrics."""
        passed = True
        reasons = []
        
        for metric_name, threshold in self.config["thresholds"].items():
            if metric_name in metrics and metrics[metric_name] < threshold:
                passed = False
                reasons.append(f"{metric_name} ({metrics[metric_name]:.4f}) below threshold ({threshold})")
                
        return {"passed": passed, "reasons": reasons, "metrics": metrics}
        
    @task
    def promote_model(self, run_id, validation_result):
        """Promote model to production if validation passes."""
        if validation_result["passed"]:
            client = mlflow.tracking.MlflowClient()
            model_name = self.config["model_name"]
            model_version = client.get_run(run_id).data.tags.get("version", "1")
            
            # Transition model to Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production"
            )
            
            return {"status": "promoted", "run_id": run_id, "version": model_version}
        else:
            return {"status": "rejected", "reasons": validation_result["reasons"]}

@flow(name="Model Validation Flow")
def model_validation_flow(run_id, test_data_path, config):
    """Orchestrate the model validation process."""
    validator = ModelValidator(config)
    model = validator.load_model(run_id)
    X_test, y_test = validator.load_test_data(test_data_path)
    metrics = validator.compute_metrics(model, X_test, y_test)
    validation_result = validator.validate_metrics(metrics)
    promotion_result = validator.promote_model(run_id, validation_result)
    return promotion_result
```

### 2.5. Model Deployment

```python
# pipelines/model_deployment/model_deployer.py
import mlflow
from prefect import task, flow
import docker
import json
import os

class ModelDeployer:
    """Model deployment."""
    
    def __init__(self, config):
        self.config = config
        
    @task
    def export_model(self, run_id):
        """Export model to deployment format."""
        import joblib
        
        # Load model
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        # Save model
        export_path = f"models/deployed/{run_id}"
        os.makedirs(export_path, exist_ok=True)
        joblib.dump(model, f"{export_path}/model.joblib")
        
        # Save metadata
        with open(f"{export_path}/metadata.json", "w") as f:
            json.dump({
                "run_id": run_id,
                "model_name": self.config["model_name"],
                "created_at": mlflow.get_run(run_id).info.start_time,
                "config": self.config
            }, f)
            
        return export_path
        
    @task
    def build_container(self, model_path):
        """Build Docker container for model."""
        # Create Dockerfile
        dockerfile = f"""
        FROM python:3.9-slim
        
        WORKDIR /app
        
        COPY {model_path}/model.joblib /app/model.joblib
        COPY {model_path}/metadata.json /app/metadata.json
        COPY app/api.py /app/api.py
        COPY requirements.txt /app/requirements.txt
        
        RUN pip install -r requirements.txt
        
        EXPOSE 8000
        
        CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile)
            
        # Build container
        client = docker.from_env()
        image, logs = client.images.build(
            path=".",
            tag=f"{self.config['model_name']}:{self.config['version']}",
            rm=True
        )
        
        return image.tags[0]
        
    @task
    def deploy_to_environment(self, image_tag):
        """Deploy container to environment."""
        if self.config["deployment_target"] == "kubernetes":
            self._deploy_to_kubernetes(image_tag)
        elif self.config["deployment_target"] == "docker":
            self._deploy_to_docker(image_tag)
        else:
            raise ValueError(f"Unsupported deployment target: {self.config['deployment_target']}")
            
        return {"status": "deployed", "image": image_tag}
        
    def _deploy_to_docker(self, image_tag):
        """Deploy to Docker."""
        client = docker.from_env()
        container = client.containers.run(
            image_tag,
            detach=True,
            ports={"8000/tcp": 8000},
            name=f"{self.config['model_name']}-{self.config['version']}"
        )
        return container.id
        
    def _deploy_to_kubernetes(self, image_tag):
        """Deploy to Kubernetes."""
        # Generate Kubernetes YAML
        model_name = self.config["model_name"]
        version = self.config["version"]
        
        deployment_yaml = f"""
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: {model_name}-{version}
        spec:
          replicas: {self.config.get("replicas", 1)}
          selector:
            matchLabels:
              app: {model_name}
              version: "{version}"
          template:
            metadata:
              labels:
                app: {model_name}
                version: "{version}"
            spec:
              containers:
              - name: {model_name}
                image: {image_tag}
                ports:
                - containerPort: 8000
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: {model_name}
        spec:
          selector:
            app: {model_name}
            version: "{version}"
          ports:
          - port: 80
            targetPort: 8000
          type: ClusterIP
        """
        
        with open(f"deployment/{model_name}-{version}.yaml", "w") as f:
            f.write(deployment_yaml)
            
        # Apply to Kubernetes
        import subprocess
        subprocess.run(["kubectl", "apply", "-f", f"deployment/{model_name}-{version}.yaml"])
        
        return f"{model_name}-{version}"

@flow(name="Model Deployment Flow")
def model_deployment_flow(run_id, config):
    """Orchestrate the model deployment process."""
    deployer = ModelDeployer(config)
    model_path = deployer.export_model(run_id)
    image_tag = deployer.build_container(model_path)
    result = deployer.deploy_to_environment(image_tag)
    return result
```

### 2.6. Monitoring and Observability

```python
# monitoring/model_monitor.py
from prefect import task, flow
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class ModelMonitor:
    """Model monitoring."""
    
    def __init__(self, config):
        self.config = config
        
    @task
    def collect_production_data(self):
        """Collect production data."""
        # Implement based on your data sources
        raise NotImplementedError
        
    @task
    def check_data_drift(self, reference_data, current_data):
        """Check for data drift."""
        from evidently.dashboard import Dashboard
        from evidently.dashboard.tabs import DataDriftTab
        
        # Create data drift report
        data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
        data_drift_dashboard.calculate(reference_data, current_data, column_mapping=None)
        
        # Save report
        os.makedirs("monitoring/reports", exist_ok=True)
        data_drift_dashboard.save(f"monitoring/reports/data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # Extract drift metrics
        drift_metrics = {
            "timestamp": datetime.now().isoformat(),
            "n_features": len(reference_data.columns),
            "n_drifted_features": sum(1 for _ in data_drift_dashboard.get_drifted_features())
        }
        
        return drift_metrics
        
    @task
    def check_model_performance(self, model, data, labels):
        """Check model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = model.predict(data)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average='weighted'),
            "recall": recall_score(labels, predictions, average='weighted'),
            "f1": f1_score(labels, predictions, average='weighted')
        }
        
        # Save metrics
        os.makedirs("monitoring/metrics", exist_ok=True)
        with open(f"monitoring/metrics/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(metrics, f)
            
        return metrics
        
    @task
    def detect_anomalies(self, metrics, window_size=30):
        """Detect anomalies in performance metrics."""
        # Load historical metrics
        historical_files = sorted(os.listdir("monitoring/metrics"))[-window_size:]
        historical_metrics = []
        
        for file in historical_files:
            with open(f"monitoring/metrics/{file}", "r") as f:
                historical_metrics.append(json.load(f))
                
        # Extract metric values
        metric_history = {
            metric: [m.get(metric, 0) for m in historical_metrics] 
            for metric in ["accuracy", "precision", "recall", "f1"]
        }
        
        # Calculate control limits (simple 3-sigma)
        control_limits = {}
        anomalies = {}
        
        for metric, values in metric_history.items():
            mean = np.mean(values)
            std = np.std(values)
            lower_limit = mean - 3 * std
            upper_limit = mean + 3 * std
            
            control_limits[metric] = {"mean": mean, "std": std, "lower": lower_limit, "upper": upper_limit}
            anomalies[metric] = not (lower_limit <= metrics[metric] <= upper_limit)
            
        return {
            "anomalies_detected": any(anomalies.values()),
            "anomalies": anomalies,
            "control_limits": control_limits,
            "current_metrics": metrics
        }
        
    @task
    def trigger_alerts(self, anomaly_results):
        """Trigger alerts if anomalies detected."""
        if anomaly_results["anomalies_detected"]:
            # Implement your alerting mechanism (email, Slack, etc.)
            print(f"ALERT: Anomalies detected: {anomaly_results['anomalies']}")
            
            # Log alert
            os.makedirs("monitoring/alerts", exist_ok=True)
            with open(f"monitoring/alerts/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                json.dump(anomaly_results, f)
                
        return anomaly_results["anomalies_detected"]

@flow(name="Model Monitoring Flow")
def model_monitoring_flow(config):
    """Orchestrate the model monitoring process."""
    monitor = ModelMonitor(config)
    production_data = monitor.collect_production_data()
    
    # Split into features and labels
    features = production_data.drop(config["target_column"], axis=1)
    labels = production_data[config["target_column"]]
    
    # Load reference data
    reference_data = pd.read_parquet(config["reference_data_path"])
    reference_features = reference_data.drop(config["target_column"], axis=1)
    
    # Load model
    import mlflow
    model = mlflow.sklearn.load_model(f"models:/{config['model_name']}/Production")
    
    # Check data drift
    drift_metrics = monitor.check_data_drift(reference_features, features)
    
    # Check model performance
    performance_metrics = monitor.check_model_performance(model, features, labels)
    
    # Detect anomalies
    anomaly_results = monitor.detect_anomalies(performance_metrics)
    
    # Trigger alerts if needed
    alert_triggered = monitor.trigger_alerts(anomaly_results)
    
    return {
        "drift_metrics": drift_metrics,
        "performance_metrics": performance_metrics,
        "anomalies": anomaly_results,
        "alert_triggered": alert_triggered
    }
```

## 3. End-to-End Pipeline Orchestration

```python
# prefect_flows/ml_pipeline.py
from prefect import flow
from pipelines.data_ingestion.data_ingestion import data_ingestion_flow
from pipelines.feature_engineering.feature_engineering import feature_engineering_flow
from pipelines.model_training.model_training import model_training_flow
from pipelines.model_evaluation.model_evaluation import model_validation_flow
from pipelines.model_deployment.model_deployment import model_deployment_flow

@flow(name="End-to-End ML Pipeline")
def ml_pipeline(config):
    """Orchestrate the end-to-end ML pipeline."""
    # Load configuration
    import yaml
    with open(config, "r") as f:
        pipeline_config = yaml.safe_load(f)
    
    # Data ingestion
    data_path = data_ingestion_flow(pipeline_config["data_ingestion"])
    
    # Feature engineering
    features_path = feature_engineering_flow(data_path, pipeline_config["feature_engineering"])
    
    # Model training
    run_id = model_training_flow(features_path, pipeline_config["model_training"])
    
    # Model validation
    validation_result = model_validation_flow(
        run_id, 
        pipeline_config["model_validation"]["test_data_path"],
        pipeline_config["model_validation"]
    )
    
    # If validation passes, deploy model
    if validation_result["status"] == "promoted":
        deployment_result = model_deployment_flow(run_id, pipeline_config["model_deployment"])
    else:
        deployment_result = {"status": "skipped", "reason": "Model validation failed"}
    
    return {
        "data_path": data_path,
        "features_path": features_path,
        "run_id": run_id,
        "validation_result": validation_result,
        "deployment_result": deployment_result
    }

# Deploy the flow
if __name__ == "__main__":
    from prefect.deployments import Deployment
    
    deployment = Deployment.build_from_flow(
        flow=ml_pipeline,
        name="ml-pipeline-deployment",
        parameters={"config": "config/pipeline_config.yaml"},
        schedule=None  # Set your desired schedule
    )
    
    deployment.apply()
```

## 4. Configuration Management

Example configuration file structure:

```yaml
# config/pipeline_config.yaml
data_ingestion:
  source_type: "postgresql"
  connection_string: "${DB_CONNECTION_STRING}"
  query: "SELECT * FROM customers WHERE date >= '2023-01-01'"
  output_path: "data/raw/customers.parquet"

feature_engineering:
  feature_group: "customer_features"
  version: "1.0.0"
  transformations:
    - name: "numerical_scaling"
      method: "standard_scaler"
      columns: ["age", "income", "tenure"]
    - name: "categorical_encoding"
      method: "one_hot_encoder"
      columns: ["gender", "location", "segment"]
  output_path: "data/features/customer_features_v1.parquet"

model_training:
  experiment_name: "customer_churn_prediction"
  model_name: "churn_predictor"
  model_type: "random_forest"
  target_column: "churn"
  test_size: 0.2
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    random_state: 42

model_validation:
  model_name: "churn_predictor"
  test_data_path: "data/features/customer_features_test.parquet"
  target_column: "churn"
  thresholds:
    accuracy: 0.85
    precision: 0.80
    recall: 0.75
    f1: 0.80

model_deployment:
  model_name: "churn_predictor"
  version: "1.0.0"
  deployment_target: "kubernetes"
  replicas: 2
  resources:
    requests:
      cpu: "500m"
      memory: "512Mi"
    limits:
      cpu: "1000m"
      memory: "1Gi"

monitoring:
  model_name: "churn_predictor"
  reference_data_path: "data/features/customer_features_v1.parquet"
  target_column: "churn"
  monitoring_interval: "1h"
  alert_channels:
    - type: "slack"
      webhook: "${SLACK_WEBHOOK_URL}"
    - type: "email"
      recipients: ["data-science-team@example.com"]
```

## 5. Installation and Dependencies

```toml
# pyproject.toml
[tool.poetry]
name = "mlops-framework"
version = "0.1.0"
description = "A framework for ML pipelines using Prefect"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
prefect = "^2.0.0"
pandas = "^1.5.0"
scikit-learn = "^1.0.0"
mlflow = "^2.0.0"
dvc = "^2.0.0"
evidently = "^0.2.0"
fastapi = "^0.85.0"
uvicorn = "^0.18.0"
pydantic = "^1.9.0"
docker = "^6.0.0"
pyyaml = "^6.0.0"
python-dotenv = "^0.21.0"
joblib = "^1.1.0"
psycopg2-binary = "^2.9.3"

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"
pytest-cov = "^3.0.0"
black = "^22.8.0"
isort = "^5.10.0"
flake8 = "^5.0.0"
mypy = "^0.971"
jupyter = "^1.0.0"
notebook = "^6.4.12"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

## 6. Service APIs and Integration Points

### 6.1. Model Serving API

```python
# app/api.py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import time
from datetime import datetime

# Load model and metadata
model = joblib.load("/app/model.joblib")

with open("/app/metadata.json", "r") as f:
    metadata = json.load(f)

app = FastAPI(title=f"Model API: {metadata['model_name']}")

class PredictionInput(BaseModel):
    features: List[Dict[str, Any]]
    
class PredictionResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]]
    model_info: Dict[str, Any]
    prediction_time: float

@app.get("/")
def root():
    """Root endpoint with model information."""
    return {
        "model_name": metadata["model_name"],
        "run_id": metadata["run_id"],
        "created_at": metadata["created_at"],
        "version": metadata.get("config", {}).get("version", "1.0.0")
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    """Make predictions."""
    start_time = time.time()
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.features)
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df).tolist()
        
        prediction_time = time.time() - start_time
        
        # Log prediction for monitoring
        os.makedirs("/app/logs", exist_ok=True)
        with open(f"/app/logs/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl", "a") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "features": input_data.features,
                "predictions": predictions,
                