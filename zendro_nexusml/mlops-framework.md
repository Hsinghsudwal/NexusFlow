# MLOps Framework Implementation

## Architecture Overview

The framework consists of modular components that can be used independently or as a cohesive system:

```
                   ┌─────────────────┐
                   │                 │
                   │  Data Sources   │
                   │                 │
                   └────────┬────────┘
                            │
                            ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│                 │  │                 │  │                 │
│  Data Version   │◄─┤ Data Management │──┤  Feature Store  │
│  Control (DVC)  │  │                 │  │    (Feast)      │
│                 │  └────────┬────────┘  │                 │
└─────────────────┘           │           └─────────────────┘
                            │
                            ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│                 │  │                 │  │                 │
│   Experiment    │◄─┤   Pipeline      │──┤     CI/CD       │
│ Tracking (MLflow)│  │Orchestration   │  │  (GitHub Actions)│
│                 │  │  (Airflow)      │  │                 │
└─────────────────┘  └────────┬────────┘  └─────────────────┘
                            │
                            ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│                 │  │                 │  │                 │
│ Model Registry  │◄─┤   Deployment    │──┤   Monitoring    │
│    (MLflow)     │  │ (Kubeflow/BentoML)│  │ (Prometheus/Grafana)│
│                 │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Core Components Implementation

### 1. Data Management Layer

#### Data Versioning with DVC
```python
# Example: Initializing and tracking data with DVC
import os
import subprocess

def initialize_dvc(project_dir):
    """Initialize DVC in the project directory."""
    os.chdir(project_dir)
    subprocess.run(["dvc", "init"])
    
def track_dataset(data_path, description):
    """Add a dataset to DVC tracking."""
    subprocess.run(["dvc", "add", data_path])
    subprocess.run(["git", "add", f"{data_path}.dvc"])
    subprocess.run(["git", "commit", "-m", f"Add dataset: {description}"])
```

#### Feature Store with Feast
```python
# example_feature_repo/feature_store.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Define an entity for our features
customer = Entity(name="customer_id", value_type=ValueType.INT64)

# Define a data source
customer_data_source = FileSource(
    path="path/to/customer_features.parquet",
    event_timestamp_column="event_timestamp",
)

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=30),
    features=[
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="total_purchases", dtype=ValueType.INT64),
        Feature(name="avg_purchase_value", dtype=ValueType.FLOAT),
    ],
    batch_source=customer_data_source,
)
```

### 2. Pipeline Components

Create a modular pipeline structure with configurable components:

```python
# mlops_framework/components/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import yaml
import os

class PipelineComponent(ABC):
    """Base class for all pipeline components."""
    
    def __init__(self, name: str, config_path: Optional[str] = None):
        self.name = name
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    @abstractmethod
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the component logic."""
        pass
    
    def save_artifacts(self, artifacts: Dict[str, Any], output_dir: str):
        """Save component artifacts to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "component_name": self.name,
            "artifact_keys": list(artifacts.keys())
        }
        
        with open(os.path.join(output_dir, "metadata.yaml"), 'w') as f:
            yaml.dump(metadata, f)
            
        # Save individual artifacts
        for key, artifact in artifacts.items():
            # Implementation depends on artifact type
            pass
```

#### Example Data Processing Component

```python
# mlops_framework/components/data_processor.py
from .base import PipelineComponent
from typing import Dict, Any
import pandas as pd

class DataProcessor(PipelineComponent):
    """Component for data preprocessing."""
    
    def run(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input data according to configuration."""
        if not inputs or "data" not in inputs:
            raise ValueError("Input data not provided")
        
        data = inputs["data"]
        
        # Apply configured transformations
        if "drop_columns" in self.config:
            data = data.drop(columns=self.config["drop_columns"])
            
        if "fill_na" in self.config:
            for col, value in self.config["fill_na"].items():
                data[col] = data[col].fillna(value)
                
        if "categorical_encoding" in self.config:
            for col in self.config["categorical_encoding"]:
                data = pd.get_dummies(data, columns=[col], drop_first=True)
        
        return {"processed_data": data}
```

### 3. Experiment Tracking with MLflow

```python
# mlops_framework/tracking.py
import mlflow
from typing import Dict, Any

class ExperimentTracker:
    """Wrapper for MLflow experiment tracking."""
    
    def __init__(self, experiment_name: str, tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            self.experiment_id = mlflow.create_experiment(experiment_name)
    
    def start_run(self, run_name: str = None):
        """Start a new tracking run."""
        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the current run."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to the current run."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifacts(self, artifact_paths: Dict[str, str]):
        """Log artifacts to the current run."""
        for name, path in artifact_paths.items():
            mlflow.log_artifact(path)
    
    def end_run(self):
        """End the current tracking run."""
        mlflow.end_run()
```

### 4. Pipeline Orchestration with Airflow

```python
# example_airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from mlops_framework.components.data_processor import DataProcessor
from mlops_framework.components.feature_engineer import FeatureEngineer
from mlops_framework.components.model_trainer import ModelTrainer
from mlops_framework.components.model_evaluator import ModelEvaluator

# Pipeline configuration
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML model training pipeline',
    schedule_interval=timedelta(days=1),
)

# Task functions
def process_data(**kwargs):
    processor = DataProcessor("data_processor", "configs/data_processor.yaml")
    # Load data
    input_data = pd.read_csv("data/raw/data.csv")
    # Process data
    result = processor.run({"data": input_data})
    # Save result for next task
    result["processed_data"].to_parquet("data/processed/processed_data.parquet")

def engineer_features(**kwargs):
    engineer = FeatureEngineer("feature_engineer", "configs/feature_engineer.yaml")
    # Load processed data
    processed_data = pd.read_parquet("data/processed/processed_data.parquet")
    # Engineer features
    result = engineer.run({"data": processed_data})
    # Save result for next task
    result["featured_data"].to_parquet("data/featured/featured_data.parquet")

def train_model(**kwargs):
    trainer = ModelTrainer("model_trainer", "configs/model_trainer.yaml")
    # Load featured data
    featured_data = pd.read_parquet("data/featured/featured_data.parquet")
    # Split data
    X = featured_data.drop("target", axis=1)
    y = featured_data["target"]
    # Train model
    result = trainer.run({"X": X, "y": y})
    # Register model with MLflow

def evaluate_model(**kwargs):
    evaluator = ModelEvaluator("model_evaluator", "configs/model_evaluator.yaml")
    # Load featured data and model
    featured_data = pd.read_parquet("data/featured/featured_data.parquet")
    # Evaluate model
    result = evaluator.run({"data": featured_data})
    # Log evaluation results

# Define tasks
t1 = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t4 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Define dependencies
t1 >> t2 >> t3 >> t4
```

### 5. Model Registry and Deployment

```python
# mlops_framework/deployment.py
import mlflow
from mlflow.tracking import MlflowClient
import os
from typing import Dict, Any

class ModelDeployer:
    """Handle model registration and deployment."""
    
    def __init__(self, model_name: str, tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
    
    def register_model(self, run_id: str, model_path: str) -> str:
        """Register a model from a specific run."""
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{model_path}",
            name=self.model_name
        )
        return result.version
    
    def transition_model(self, version: str, stage: str):
        """Transition a model to a new stage (Staging/Production/Archived)."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
    
    def deploy_model(self, version: str, deployment_config: Dict[str, Any]):
        """Deploy a specific model version."""
        # Implementation depends on deployment target (Kubernetes, BentoML, etc.)
        # Example for BentoML deployment:
        if deployment_config.get("type") == "bentoml":
            # Create BentoML service
            # Package model
            # Deploy to target platform
            pass
```

### 6. CI/CD Integration

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI/CD

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'configs/**'
      - 'data/raw/**'
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly pipeline run

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      - name: Test
        run: pytest tests/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run training pipeline
        run: python -m src.pipelines.training_pipeline
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      
  deploy-staging:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Deploy to staging
        run: python -m src.deployment.deploy --environment staging
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

### 7. Monitoring Setup

```python
# mlops_framework/monitoring.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import json
import prometheus_client

class ModelMonitor:
    """Monitor deployed models for data drift and performance."""
    
    def __init__(self, model_name: str, version: str, feature_columns: List[str]):
        self.model_name = model_name
        self.version = version
        self.feature_columns = feature_columns
        
        # Set up metrics
        self.prediction_counter = prometheus_client.Counter(
            'model_predictions_total', 
            'Total number of predictions', 
            ['model_name', 'version']
        )
        self.prediction_latency = prometheus_client.Histogram(
            'model_prediction_latency_seconds', 
            'Prediction latency in seconds',
            ['model_name', 'version']
        )
        self.feature_drift = prometheus_client.Gauge(
            'model_feature_drift', 
            'Feature drift score',
            ['model_name', 'version', 'feature']
        )
        
    def log_prediction(self, prediction_data: Dict[str, Any], latency: float):
        """Log a single prediction event."""
        self.prediction_counter.labels(
            model_name=self.model_name, 
            version=self.version
        ).inc()
        
        self.prediction_latency.labels(
            model_name=self.model_name, 
            version=self.version
        ).observe(latency)
        
    def calculate_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """Calculate and log feature drift between reference and current data."""
        for feature in self.feature_columns:
            if feature in reference_data.columns and feature in current_data.columns:
                # Simple distribution difference metric (can be replaced with more sophisticated approaches)
                reference_mean = reference_data[feature].mean()
                current_mean = current_data[feature].mean()
                
                if reference_mean != 0:
                    drift_score = abs((current_mean - reference_mean) / reference_mean)
                else:
                    drift_score = abs(current_mean - reference_mean)
                
                self.feature_drift.labels(
                    model_name=self.model_name,
                    version=self.version,
                    feature=feature
                ).set(drift_score)
```

## Project Structure

```
mlops_framework/
├── configs/
│   ├── data_processor.yaml
│   ├── feature_engineer.yaml
│   ├── model_trainer.yaml
│   └── model_evaluator.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── featured/
├── mlops_framework/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── data_processor.py
│   │   ├── feature_engineer.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── deployment.py
│   ├── monitoring.py
│   └── tracking.py
├── pipelines/
│   ├── __init__.py
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_components.py
│   └── test_pipelines.py
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── README.md
└── requirements.txt
```
