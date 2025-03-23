# MLOps Customer Churn Project

This document outlines an end-to-end MLOps pipeline for customer churn prediction built from scratch with Prefect, LocalStack, Docker, and other modern tools. The architecture follows the principles of ZenML and Kedro without using those libraries directly.

## Project Structure
```
mlops-churn/
├── config/
│   ├── model_config.yaml            # Model hyperparameters
│   ├── pipeline_config.yaml         # Pipeline configuration
│   └── monitoring_config.yaml       # Monitoring thresholds and settings
├── data/
│   ├── raw/                         # Raw data storage
│   ├── processed/                   # Processed data
│   └── features/                    # Feature stores
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Load raw data
│   │   ├── data_validation.py       # Validate input data
│   │   └── feature_engineering.py   # Create features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                 # Model training 
│   │   ├── evaluate.py              # Model evaluation
│   │   ├── predict.py               # Make predictions
│   │   └── model_registry.py        # Model versioning and registry
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py         # Data processing pipeline
│   │   ├── training_pipeline.py     # Model training pipeline
│   │   ├── evaluation_pipeline.py   # Model evaluation pipeline
│   │   └── deployment_pipeline.py   # Model deployment pipeline
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── data_drift.py            # Data drift detection
│   │   ├── model_performance.py     # Model performance monitoring
│   │   └── alerting.py              # Alert system
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── model_server.py          # Model serving code
│   │   └── api.py                   # REST API for model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration loader
│   │   ├── logging_utils.py         # Logging utilities
│   │   └── metadata.py              # Metadata tracking
│   └── db/
│       ├── __init__.py
│       ├── local_db.py              # Local database utilities
│       └── migrations/              # Database migrations
├── notebooks/
│   └── exploratory_analysis.ipynb   # EDA notebook
├── tests/
│   ├── __init__.py
│   ├── test_data.py                 # Data tests
│   ├── test_models.py               # Model tests
│   └── test_pipelines.py            # Pipeline tests
├── docker/
│   ├── Dockerfile                   # Main application Dockerfile
│   ├── docker-compose.yml           # Compose file for local development
│   └── localstack.yml               # LocalStack configuration
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI pipeline
│       └── cd.yml                   # CD pipeline
├── main.py                          # Main entry point for running pipelines
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Core Components Implementation

Below is the implementation of key files in the project:

### 1. `main.py` - Entry Point

```python
#!/usr/bin/env python
import os
import argparse
import logging
from datetime import datetime

from prefect import Flow
from src.utils.config import load_config
from src.pipelines.data_pipeline import create_data_pipeline
from src.pipelines.training_pipeline import create_training_pipeline
from src.pipelines.evaluation_pipeline import create_evaluation_pipeline
from src.pipelines.deployment_pipeline import create_deployment_pipeline
from src.utils.logging_utils import setup_logging
from src.monitoring.data_drift import trigger_drift_check
from src.db.local_db import initialize_db

# Setup logging
logger = setup_logging()

def parse_args():
    parser = argparse.ArgumentParser(description='MLOps Churn Pipeline')
    parser.add_argument('--pipeline', type=str, default='full',
                        choices=['data', 'train', 'evaluate', 'deploy', 'retrain', 'full'],
                        help='Pipeline to run')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                        help='Path to pipeline configuration')
    return parser.parse_args()

def run_pipeline(pipeline_name, config_path):
    """Run the specified pipeline with the given configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Initialize the database
    initialize_db()
    
    # Setup artifact directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join("artifacts", run_id)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Create flow based on requested pipeline
    if pipeline_name == 'data' or pipeline_name == 'full':
        with Flow("data-pipeline") as flow:
            data_pipeline = create_data_pipeline(config, artifact_dir)
        flow.run()
        logger.info(f"Data pipeline completed. Artifacts saved to {artifact_dir}")
    
    if pipeline_name == 'train' or pipeline_name == 'full':
        with Flow("training-pipeline") as flow:
            training_pipeline = create_training_pipeline(config, artifact_dir)
        flow.run()
        logger.info(f"Training pipeline completed. Model saved to {artifact_dir}")
    
    if pipeline_name == 'evaluate' or pipeline_name == 'full':
        with Flow("evaluation-pipeline") as flow:
            evaluation_pipeline = create_evaluation_pipeline(config, artifact_dir)
        flow.run()
        logger.info(f"Evaluation pipeline completed. Metrics saved to {artifact_dir}")
    
    if pipeline_name == 'deploy' or pipeline_name == 'full':
        with Flow("deployment-pipeline") as flow:
            deployment_pipeline = create_deployment_pipeline(config, artifact_dir)
        flow.run()
        logger.info("Deployment pipeline completed. Model deployed to production")
    
    if pipeline_name == 'retrain':
        # Check for data drift before retraining
        drift_detected = trigger_drift_check(config)
        if drift_detected:
            logger.info("Data drift detected. Starting retraining...")
            with Flow("retraining-pipeline") as flow:
                training_pipeline = create_training_pipeline(config, artifact_dir, is_retraining=True)
            flow.run()
            
            with Flow("redeployment-pipeline") as flow:
                deployment_pipeline = create_deployment_pipeline(config, artifact_dir)
            flow.run()
            logger.info("Retraining and redeployment completed")
        else:
            logger.info("No significant data drift detected. Skipping retraining.")
    
    return artifact_dir

if __name__ == "__main__":
    args = parse_args()
    artifact_dir = run_pipeline(args.pipeline, args.config)
    print(f"Pipeline execution completed. Artifacts saved to {artifact_dir}")
```

### 2. `src/pipelines/data_pipeline.py` - Data Processing Pipeline

```python
from prefect import task, Flow
import pandas as pd
import os
import joblib
from src.data.data_loader import load_data
from src.data.data_validation import validate_data
from src.data.feature_engineering import engineer_features
from src.utils.metadata import save_metadata

@task
def extract_data(config):
    """Extract data from source."""
    return load_data(config["data_source"])

@task
def validate_raw_data(data, config):
    """Validate the raw data."""
    validation_results = validate_data(data, config["schema"])
    if not validation_results["is_valid"]:
        raise ValueError(f"Data validation failed: {validation_results['errors']}")
    return data

@task
def transform_data(data, config):
    """Transform and engineer features."""
    return engineer_features(data, config["features"])

@task
def save_processed_data(data, artifact_dir):
    """Save processed data and feature metadata."""
    processed_data_path = os.path.join(artifact_dir, "processed_data.csv")
    data.to_csv(processed_data_path, index=False)
    
    # Save feature list and metadata
    feature_metadata = {
        "features": list(data.columns),
        "n_samples": len(data),
        "feature_stats": {col: {"mean": data[col].mean(), 
                               "std": data[col].std()} 
                        for col in data.columns if data[col].dtype in ['int64', 'float64']}
    }
    
    metadata_path = os.path.join(artifact_dir, "feature_metadata.json")
    save_metadata(feature_metadata, metadata_path)
    
    return processed_data_path

def create_data_pipeline(config, artifact_dir):
    """Create and return the data pipeline flow."""
    with Flow("data-pipeline") as flow:
        raw_data = extract_data(config)
        validated_data = validate_raw_data(raw_data, config)
        transformed_data = transform_data(validated_data, config)
        processed_data_path = save_processed_data(transformed_data, artifact_dir)
    
    return flow
```

### 3. `src/pipelines/training_pipeline.py` - Model Training Pipeline

```python
from prefect import task, Flow
import pandas as pd
import os
import joblib
import json
from datetime import datetime
from src.models.train import train_model
from src.utils.metadata import save_metadata
from src.db.local_db import save_training_run

@task
def load_processed_data(artifact_dir):
    """Load the processed data from the artifact directory."""
    processed_data_path = os.path.join(artifact_dir, "processed_data.csv")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
    
    return pd.read_csv(processed_data_path)

@task
def prepare_training_data(data, config):
    """Split data into training and validation sets."""
    # Split features and target
    X = data.drop(config["target_column"], axis=1)
    y = data[config["target_column"]]
    
    # Train-validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config["validation_split"], random_state=42
    )
    
    return {
        "X_train": X_train, 
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feature_names": list(X.columns)
    }

@task
def train_and_validate_model(training_data, config, artifact_dir, is_retraining=False):
    """Train the model and validate it."""
    # Train model
    model, training_metrics = train_model(
        training_data["X_train"], 
        training_data["y_train"],
        training_data["X_val"],
        training_data["y_val"],
        config["model_params"]
    )
    
    # Save model
    model_path = os.path.join(artifact_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # Save training metadata
    training_metadata = {
        "model_type": config["model_type"],
        "hyperparameters": config["model_params"],
        "features": training_data["feature_names"],
        "training_metrics": training_metrics,
        "timestamp": datetime.now().isoformat(),
        "is_retraining": is_retraining
    }
    
    metadata_path = os.path.join(artifact_dir, "training_metadata.json")
    save_metadata(training_metadata, metadata_path)
    
    # Save to local database
    save_training_run(
        model_version=artifact_dir.split("/")[-1],
        model_type=config["model_type"],
        metrics=training_metrics,
        is_retraining=is_retraining
    )
    
    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics": training_metrics
    }

def create_training_pipeline(config, artifact_dir, is_retraining=False):
    """Create and return the training pipeline flow."""
    with Flow("training-pipeline") as flow:
        data = load_processed_data(artifact_dir)
        training_data = prepare_training_data(data, config)
        training_result = train_and_validate_model(
            training_data, 
            config, 
            artifact_dir,
            is_retraining
        )
    
    return flow
```

### 4. `src/pipelines/evaluation_pipeline.py` - Model Evaluation Pipeline

```python
from prefect import task, Flow
import pandas as pd
import os
import joblib
import json
import numpy as np
from src.models.evaluate import evaluate_model, compare_models
from src.utils.metadata import load_metadata, save_metadata

@task
def load_model(artifact_dir):
    """Load the trained model from the artifact directory."""
    model_path = os.path.join(artifact_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)

@task
def load_test_data(config):
    """Load test data for evaluation."""
    test_data_path = config["test_data_path"]
    return pd.read_csv(test_data_path)

@task
def evaluate_current_model(model, test_data, config, artifact_dir):
    """Evaluate the current model on test data."""
    # Prepare test data
    X_test = test_data.drop(config["target_column"], axis=1)
    y_test = test_data[config["target_column"]]
    
    # Evaluate model
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    
    # Save evaluation results
    eval_results = {
        "metrics": evaluation_metrics,
        "threshold": config.get("prediction_threshold", 0.5),
        "data_size": len(test_data),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    eval_path = os.path.join(artifact_dir, "evaluation_results.json")
    save_metadata(eval_results, eval_path)
    
    return eval_results

@task
def find_production_model():
    """Find the current production model if it exists."""
    production_path = "models/production"
    if os.path.exists(production_path):
        model_path = os.path.join(production_path, "model.joblib")
        metadata_path = os.path.join(production_path, "evaluation_results.json")
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            return {
                "model": joblib.load(model_path),
                "metadata": load_metadata(metadata_path)
            }
    
    return None

@task
def compare_with_production(current_eval, production_model, test_data, config):
    """Compare current model with production model."""
    if production_model is None:
        return {
            "is_better": True,
            "improvement": 1.0,
            "comparison": {"current": current_eval["metrics"]}
        }
    
    # Prepare test data
    X_test = test_data.drop(config["target_column"], axis=1)
    y_test = test_data[config["target_column"]]
    
    # Compare models
    comparison_results = compare_models(
        current_model={"model": None, "metrics": current_eval["metrics"]},  # We already have metrics
        production_model={"model": production_model["model"], "metrics": production_model["metadata"]["metrics"]},
        X_test=X_test,
        y_test=y_test
    )
    
    return comparison_results

@task
def save_comparison_results(comparison_results, artifact_dir):
    """Save model comparison results."""
    comparison_path = os.path.join(artifact_dir, "model_comparison.json")
    save_metadata(comparison_results, comparison_path)
    
    return comparison_path

def create_evaluation_pipeline(config, artifact_dir):
    """Create and return the evaluation pipeline flow."""
    with Flow("evaluation-pipeline") as flow:
        model = load_model(artifact_dir)
        test_data = load_test_data(config)
        
        # Evaluate current model
        current_eval = evaluate_current_model(model, test_data, config, artifact_dir)
        
        # Find production model if exists
        production_model = find_production_model()
        
        # Compare with production model
        comparison = compare_with_production(current_eval, production_model, test_data, config)
        
        # Save comparison results
        comparison_path = save_comparison_results(comparison, artifact_dir)
    
    return flow
```

### 5. `src/pipelines/deployment_pipeline.py` - Model Deployment Pipeline

```python
from prefect import task, Flow
import os
import shutil
import json
import joblib
import docker
from src.utils.metadata import load_metadata
from src.deployment.model_server import prepare_model_server
from src.monitoring.model_performance import setup_monitoring

@task
def check_model_performance(artifact_dir, threshold=0.7):
    """Check if model performance meets the threshold for deployment."""
    comparison_path = os.path.join(artifact_dir, "model_comparison.json")
    if not os.path.exists(comparison_path):
        raise FileNotFoundError(f"Model comparison results not found at {comparison_path}")
    
    comparison = load_metadata(comparison_path)
    
    # If new model is better than production or there's no production model yet
    if comparison.get("is_better", False):
        return True
    
    # If model doesn't meet absolute performance threshold
    eval_path = os.path.join(artifact_dir, "evaluation_results.json")
    evaluation = load_metadata(eval_path)
    
    if evaluation["metrics"]["auc_roc"] < threshold:
        return False
    
    return True

@task
def deploy_model(artifact_dir, config):
    """Deploy the model to production environment."""
    production_dir = "models/production"
    os.makedirs(production_dir, exist_ok=True)
    
    # Copy model and metadata files to production directory
    source_files = [
        "model.joblib",
        "training_metadata.json",
        "evaluation_results.json",
        "feature_metadata.json"
    ]
    
    for file in source_files:
        source_path = os.path.join(artifact_dir, file)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(production_dir, file))
    
    # Update production config
    prod_config = {
        "model_version": artifact_dir.split("/")[-1],
        "deployment_timestamp": json.dumps(pd.Timestamp.now(), default=str),
        "api_endpoint": config.get("api_endpoint", "http://localhost:8000/predict"),
        "monitoring_enabled": config.get("monitoring_enabled", True)
    }
    
    with open(os.path.join(production_dir, "production_config.json"), "w") as f:
        json.dump(prod_config, f, indent=2)
    
    return production_dir

@task
def build_and_deploy_container(production_dir, config):
    """Build and deploy Docker container with the model."""
    if not config.get("use_docker", False):
        return None
    
    # Setup Docker client
    client = docker.from_env()
    
    # Build the Docker image
    image, build_logs = client.images.build(
        path="./docker",
        tag="churn-prediction:latest",
        buildargs={
            "MODEL_PATH": production_dir
        }
    )
    
    # Run the container
    container = client.containers.run(
        "churn-prediction:latest",
        detach=True,
        ports={f"{config['container_port']}/tcp": config["host_port"]},
        environment={
            "MODEL_PATH": "/app/model",
            "API_PORT": str(config["container_port"])
        },
        name="churn-prediction-service"
    )
    
    return container.id

@task
def setup_model_monitoring(production_dir, config):
    """Setup monitoring for the deployed model."""
    if not config.get("monitoring_enabled", True):
        return None
    
    # Load model metadata
    model_metadata = load_metadata(os.path.join(production_dir, "training_metadata.json"))
    feature_metadata = load_metadata(os.path.join(production_dir, "feature_metadata.json"))
    
    # Setup monitoring
    monitoring_config = setup_monitoring(
        model_metadata=model_metadata,
        feature_metadata=feature_metadata,
        monitoring_config=config.get("monitoring", {})
    )
    
    # Save monitoring configuration
    monitoring_path = os.path.join(production_dir, "monitoring_config.json")
    with open(monitoring_path, "w") as f:
        json.dump(monitoring_config, f, indent=2)
    
    return monitoring_path

def create_deployment_pipeline(config, artifact_dir):
    """Create and return the deployment pipeline flow."""
    with Flow("deployment-pipeline") as flow:
        # Check if model should be deployed
        should_deploy = check_model_performance(artifact_dir, config.get("deployment_threshold", 0.7))
        
        # Deploy model if performance is good
        production_dir = deploy_model(artifact_dir, config, upstream_tasks=[should_deploy])
        
        # Build and deploy Docker container
        container_id = build_and_deploy_container(production_dir, config)
        
        # Setup monitoring
        monitoring_path = setup_model_monitoring(production_dir, config)
    
    return flow
```

### 6. `src/models/train.py` - Model Training Implementation

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, X_val, y_val, params):
    """
    Train a model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Model parameters including model_type and hyperparameters
        
    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    model_type = params.get("model_type", "random_forest")
    logger.info(f"Training {model_type} model")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "liblinear"),
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= params.get("threshold", 0.5)).astype(int)
    
    # Calculate metrics
    metrics = {
        "auc_roc": roc_auc_score(y_val, y_pred_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred)
    }
    
    logger.info(f"Validation metrics: {metrics}")
    
    return model, metrics
```

### 7. `src/models/evaluate.py` - Model Evaluation Implementation

```python
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Prediction threshold
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    logger.info(f"Test metrics: {metrics}")
    
    return metrics

def compare_models(current_model, production_model, X_test, y_test):
    """
    Compare current model with production model.
    
    Args:
        current_model: Dictionary with current model and/or metrics
        production_model: Dictionary with production model and/or metrics
        X_test: Test features
        y_test: Test target
        
    Returns:
        comparison: Comparison results
    """
    # If metrics are already provided, use them
    current_metrics = current_model.get("metrics")
    production_metrics = production_model.get("metrics")
    
    # If metrics are not provided, evaluate the models
    if current_metrics is None and current_model.get("model") is not None:
        current_metrics = evaluate_model(current_model["model"], X_test, y_test)
    
    if production_metrics is None and production_model.get("model") is not None:
        production_metrics = evaluate_model(production_model["model"], X_test, y_test)
    
    # Compare the primary metric (AUC-ROC)
    current_auc = current_metrics.get("auc_roc", 0)
    production_auc = production_metrics.get("auc_roc", 0) if production_metrics else 0
    
    # Calculate relative improvement
    improvement = (current_auc - production_auc) / max(production_auc, 0.0001)
    
    # Determine if current model is better
    is_better = current_auc > production_auc
    
    comparison = {
        "is_better": is_better,
        "improvement": improvement,
        "comparison": {
            "current": current_metrics,
            "production": production_metrics
        }
    }
    
    logger.info(f"Model comparison: current model is {'better' if is_better else 'worse'} " +
               f"by {improvement:.2%} AUC-ROC")
    
    return comparison
```

### 8. `src/monitoring/data_drift.py` - Data Drift Detection Implementation

```python
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import json
from src.utils.metadata import load_metadata, save_metadata
import logging

logger = logging.getLogger(__name__)

def detect_data_drift(reference_data, current_data, categorical_columns=None, numerical_columns=None):
    """
    Detect data drift between reference and current data.
    
    Args:
        reference_data: Reference dataset (baseline)
        current_data: Current dataset to check for drift
        categorical_columns: List of categorical columns
        numerical_columns: List of numerical columns
        
    Returns:
        drift_results: Dictionary with drift detection results
    """
    drift_results = {
        "drift_detected": False,
        "drifted_features": [],
        "drift_scores": {}
    }
    
    # If no columns specified, infer from data types
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = reference_data.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Check drift in numerical columns using KS test
    for col in numerical_columns:
        if col in reference_data.columns and col in current_data.columns:
            # Remove nulls for the test
            ref_col = reference_data[col].dropna()
            curr_col = current_data[col].dropna()
            
            if len(ref_col) > 5 and len(curr_col) > 5:  # Ensure enough data for test
                ks_stat, p_value = ks_2samp(ref_col, curr_col)
                drift_results["drift_scores"][col] = {
                    "statistic": ks_stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
                
                if p_value < 0.05:
                    drift_results["drifted_features"].append(col)
                    drift_results["drift_detected"] = True
    
    # Check drift in categorical columns using chi-square or distribution difference
    for col in categorical_columns:
        if col in reference_data.columns and col in current_data.columns:
            # Calculate distribution
            ref_dist = reference_data[col].value_counts(normalize=True).to_dict()
            curr_dist = current_data[col].value_counts(normalize=True).to_dict()
            
            # Calculate Jensen-Shannon distance or another distribution difference
            js_distance = calculate_distribution_difference(ref_dist, curr_dist)
            
            drift_results["drift_scores"][col] = {
                "distance": js_distance,
                "drift_detected": js