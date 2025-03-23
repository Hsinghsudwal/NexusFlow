# MLOps End-to-End System for Customer Churn Prediction

## Project Structure

```
customer-churn-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI workflow
│       └── cd.yml                   # CD workflow
├── data/
│   ├── raw/                         # Raw data samples
│   └── processed/                   # Processed data samples
├── kubernetes/
│   ├── base/                        # Base Kubernetes configurations
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── kustomization.yaml
│   ├── overlays/                    # Environment-specific configurations
│   │   ├── dev/
│   │   │   └── kustomization.yaml
│   │   └── prod/
│   │       └── kustomization.yaml
│   └── kubeflow/                    # Kubeflow-specific configurations
│       └── pipeline-definition.yaml
├── notebooks/                       # Exploratory data analysis notebooks
│   └── churn_analysis.ipynb
├── pipelines/
│   ├── components/                  # Kubeflow pipeline components
│   │   ├── data_extraction/
│   │   │   ├── component.yaml
│   │   │   └── src/
│   │   │       └── extract.py
│   │   ├── data_processing/
│   │   │   ├── component.yaml
│   │   │   └── src/
│   │   │       └── process.py
│   │   ├── model_training/
│   │   │   ├── component.yaml
│   │   │   └── src/
│   │   │       └── train.py
│   │   ├── model_evaluation/
│   │   │   ├── component.yaml
│   │   │   └── src/
│   │   │       └── evaluate.py
│   │   └── model_deployment/
│   │       ├── component.yaml
│   │       └── src/
│   │           └── deploy.py
│   └── pipeline.py                  # Kubeflow pipeline definition
├── src/
│   ├── api/                         # Model serving API
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── predictions.py
│   │   └── schemas.py
│   ├── data/                        # Data processing modules
│   │   ├── __init__.py
│   │   ├── extraction.py
│   │   └── processing.py
│   ├── models/                      # ML model code
│   │   ├── __init__.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── monitoring/                  # Monitoring code
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── drift_detection.py
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       └── aws.py
├── tests/                           # Unit and integration tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_models.py
├── docker/                          # Docker-related files
│   ├── api/
│   │   └── Dockerfile
│   ├── training/
│   │   └── Dockerfile
│   └── notebook/
│       └── Dockerfile
├── docker-compose.yml              # Local development setup
├── localstack/                     # LocalStack configuration
│   └── init-aws.sh
├── skaffold.yaml                   # Skaffold configuration for K8s
├── requirements.txt                # Project dependencies
├── setup.py                        # Package setup
└── README.md                       # Project documentation
```

## Key Components Implementation

### 1. Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL database for local development
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: customer_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # LocalStack for AWS service emulation
  localstack:
    image: localstack/localstack:latest
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3,lambda,sqs,iam,cloudwatch
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - ./localstack:/docker-entrypoint-initaws.d
      - localstack_data:/tmp/localstack/data

  # Jupyter notebook for data exploration
  jupyter:
    build:
      context: .
      dockerfile: docker/notebook/Dockerfile
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
    depends_on:
      - postgres
      - localstack

  # Model API development service
  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@postgres:5432/customer_db
      - AWS_ENDPOINT_URL=http://localstack:4566
      - AWS_ACCESS_KEY_ID=test
      - AWS_SECRET_ACCESS_KEY=test
      - MODEL_BUCKET=models
    depends_on:
      postgres:
        condition: service_healthy
      localstack:
        condition: service_started

  # MLflow tracking server for experiment tracking
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://localstack:4566
      - AWS_ACCESS_KEY_ID=test
      - AWS_SECRET_ACCESS_KEY=test
    command: mlflow server --host 0.0.0.0 --backend-store-uri postgresql://postgres:postgres123@postgres:5432/customer_db --default-artifact-root s3://mlflow-artifacts/
    depends_on:
      postgres:
        condition: service_healthy
      localstack:
        condition: service_started

  # MinIO for model storage (alternative to LocalStack S3)
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  localstack_data:
  minio_data:
```

### 2. LocalStack Initialization Script

```bash
# localstack/init-aws.sh
#!/bin/bash

# Create S3 buckets
awslocal s3 mb s3://customer-data
awslocal s3 mb s3://models
awslocal s3 mb s3://mlflow-artifacts

# Create SQS queues
awslocal sqs create-queue --queue-name model-retraining-queue

# Upload sample data (if exists)
if [ -d "/docker-entrypoint-initaws.d/sample-data" ]; then
  awslocal s3 cp /docker-entrypoint-initaws.d/sample-data s3://customer-data/ --recursive
fi

echo "LocalStack AWS resources initialized successfully!"
```

### 3. Data Extraction Component

```python
# pipelines/components/data_extraction/src/extract.py
import pandas as pd
import boto3
import os
import argparse
from sqlalchemy import create_engine

def extract_data(db_host, db_name, db_user, db_password, output_path):
    """
    Extract customer data from a PostgreSQL database and save it to a specified path.
    
    Args:
        db_host: Database host
        db_name: Database name
        db_user: Database username
        db_password: Database password
        output_path: Path to save the extracted data
    """
    print(f"Connecting to database {db_name} at {db_host}")
    
    # Create a database connection
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    engine = create_engine(connection_string)
    
    # Example query - adapt to your schema
    query = """
    SELECT 
        customer_id,
        tenure,
        contract_type,
        monthly_charges,
        total_charges,
        internet_service,
        online_security,
        tech_support,
        streaming_tv,
        streaming_movies,
        payment_method,
        CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END as churn
    FROM customers
    WHERE created_at >= NOW() - INTERVAL '3 months'
    """
    
    # Execute the query
    print("Executing SQL query to extract customer data")
    df = pd.read_sql(query, engine)
    
    # Print a summary of the data
    print(f"Extracted {len(df)} customer records")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Save data to the output path
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Also save metadata about the extraction
    metadata = {
        "num_records": len(df),
        "churn_rate": float(df['churn'].mean()),
        "extraction_time": pd.Timestamp.now().isoformat(),
        "feature_columns": list(df.columns)
    }
    
    metadata_path = os.path.join(os.path.dirname(output_path), "metadata.json")
    pd.Series(metadata).to_json(metadata_path)
    print(f"Metadata saved to {metadata_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract customer data from database')
    parser.add_argument('--db-host', type=str, required=True, help='Database host')
    parser.add_argument('--db-name', type=str, required=True, help='Database name')
    parser.add_argument('--db-user', type=str, required=True, help='Database username')
    parser.add_argument('--db-password', type=str, required=True, help='Database password')
    parser.add_argument('--output-path', type=str, required=True, help='Output path for extracted data')
    
    args = parser.parse_args()
    
    extract_data(
        args.db_host,
        args.db_name,
        args.db_user,
        args.db_password,
        args.output_path
    )
```

### 4. Data Processing Component

```python
# pipelines/components/data_processing/src/process.py
import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def process_data(input_path, output_dir, preprocessor_path=None):
    """
    Process raw customer data for model training.
    
    Args:
        input_path: Path to the input CSV file
        output_dir: Directory to save processed data
        preprocessor_path: Path to save the preprocessor pipeline
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Split features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Drop identifiers
    if 'customer_id' in X.columns:
        X = X.drop('customer_id', axis=1)
    
    # Define feature types
    numeric_features = [
        'tenure', 
        'monthly_charges', 
        'total_charges'
    ]
    
    categorical_features = [
        'contract_type', 
        'internet_service', 
        'online_security',
        'tech_support', 
        'streaming_tv', 
        'streaming_movies', 
        'payment_method'
    ]
    
    # Handle numeric features that are actually strings
    for col in numeric_features:
        if X[col].dtype == 'object':
            # Remove non-numeric characters and convert to float
            X[col] = pd.to_numeric(X[col].str.replace('[^0-9.]', '', regex=True), errors='coerce')
    
    # Handle missing values
    print(f"Missing values before imputation: {X.isnull().sum().sum()}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')
    
    # Fit and transform the feature data
    print("Fitting preprocessor and transforming data")
    X_processed = preprocessor.fit_transform(X)
    
    # Save the preprocessor
    if preprocessor_path:
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
    
    # Save the processed data
    output_features_path = os.path.join(output_dir, "X_processed.npy")
    output_target_path = os.path.join(output_dir, "y.npy")
    
    np.save(output_features_path, X_processed)
    np.save(output_target_path, y.values)
    
    print(f"Processed features saved to {output_features_path}")
    print(f"Target values saved to {output_target_path}")
    
    # Save column names and feature metadata for later interpretability
    feature_names = (
        numeric_features +
        [f"{col}_{cat}" for col in categorical_features for cat in preprocessor.transformers_[1][1]['onehot'].categories_[preprocessor.transformers_[1][2].index(col)]]
    )
    
    feature_metadata = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'num_numeric': len(numeric_features),
        'num_categorical': sum([len(cats) for cats in preprocessor.transformers_[1][1]['onehot'].categories_]),
        'original_categorical_features': categorical_features,
        'original_numeric_features': numeric_features
    }
    
    feature_metadata_path = os.path.join(output_dir, "feature_metadata.json")
    pd.Series(feature_metadata).to_json(feature_metadata_path)
    print(f"Feature metadata saved to {feature_metadata_path}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process customer data for modeling')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--preprocessor-path', type=str, help='Path to save the preprocessor')
    
    args = parser.parse_args()
    
    process_data(
        args.input_path,
        args.output_dir,
        args.preprocessor_path
    )
```

### 5. Model Training Component

```python
# pipelines/components/model_training/src/train.py
import numpy as np
import pandas as pd
import os
import joblib
import json
import argparse
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(
    features_path, 
    target_path, 
    model_dir, 
    model_type='random_forest',
    tracking_uri=None,
    experiment_name='churn_prediction',
    run_name=None
):
    """
    Train a machine learning model for customer churn prediction.
    
    Args:
        features_path: Path to the processed features numpy array
        target_path: Path to the target values numpy array
        model_dir: Directory to save the trained model
        model_type: Type of model to train ('random_forest', 'gradient_boosting', 'logistic')
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        run_name: MLflow run name
    """
    # Load the processed data
    print(f"Loading data from {features_path} and {target_path}")
    X = np.load(features_path)
    y = np.load(target_path)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Set up MLflow tracking
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    run_name = run_name or f"churn_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log data information
        mlflow.log_param("data_shape", X.shape)
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("validation_size", X_val.shape[0])
        mlflow.log_param("churn_rate", np.mean(y))
        
        # Initialize model based on type
        if model_type == 'random_forest':
            # Define parameter grid for RandomForest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Initialize base model
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        elif model_type == 'gradient_boosting':
            # Define parameter grid for GradientBoosting
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            
            # Initialize base model
            base_model = GradientBoostingClassifier(random_state=42)
            
        elif model_type == 'logistic':
            # Define parameter grid for LogisticRegression
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': [None, 'balanced']
            }
            
            # Initialize base model
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Log model type and hyperparameters
        mlflow.log_param("model_type", model_type)
        for param, values in param_grid.items():
            mlflow.log_param(f"param_grid_{param}", values)
        
        # Perform grid search with cross-validation
        print(f"Performing grid search for {model_type} model")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        # Make predictions on validation set
        y_pred = best_model.predict(X_val)
        y_prob = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_prob)
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Save model locally
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")
        metrics_path = os.path.join(model_dir, "metrics.json")
        
        joblib.dump(best_model, model_path)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, "model")
        
        # Return paths and run ID for the pipeline
        return {
            'model_path': model_path,
            'metrics_path': metrics_path,
            'run_id': run.info.run_id
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a churn prediction model')
    parser.add_argument('--features-path', type=str, required=True, help='Path to processed features')
    parser.add_argument('--target-path', type=str, required=True, help='Path to target values')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--model-type', type=str, default='random_forest', 
                        choices=['random_forest', 'gradient_boosting', 'logistic'], 
                        help='Type of model to train')
    parser.add_argument('--tracking-uri', type=str, help='MLflow tracking URI')
    parser.add_argument('--experiment-name', type=str, default='churn_prediction', help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, help='MLflow run name')
    
    args = parser.parse_args()
    
    train_model(
        args.features_path,
        args.target_path,
        args.model_dir,
        args.model_type,
        args.tracking_uri,
        args.experiment_name,
        args.run_name
    )
```

### 6. Model Evaluation Component

```python
# pipelines/components/model_evaluation/src/evaluate.py
import numpy as np
import pandas as pd
import json
import os
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score
)

def evaluate_model(
    model_path, 
    features_path, 
    target_path, 
    output_dir,
    deploy_threshold=0.75
):
    """
    Evaluate a trained churn prediction model and generate evaluation artifacts.
    
    Args:
        model_path: Path to the trained model file
        features_path: Path to the processed features numpy array
        target_path: Path to the target values numpy array
        output_dir: Directory to save evaluation results
        deploy_threshold: Threshold for model to be considered deployable (ROC AUC)
    
    Returns:
        Dictionary with evaluation metrics and deployment decision
    """
    # Load model and data
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading data from {features_path} and {target_path}")
    X = np.load(features_path)
    y = np.load(target_path)
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    cm = confusion_matrix(y, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Churned', 'Churned'], rotation=45)
    plt.yticks(tick_marks, ['Not Churned', 'Churned'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    
    # Save Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_path)
    plt.close()
    
    # Calculate class distribution for the dataset
    class_distribution = {
        'not_churned': int((y == 0).sum()),
        'churned': int((y == 1).sum()),
        'churn_rate': float((y == 1).mean())
    }
    
    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0,
        'roc_auc': float(roc_auc),
        'avg_precision': float(avg_precision),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'class_distribution': class_distribution,
        'evaluation_data_size': len(y)
    }
    
    # Make deployment decision based on threshold
    deploy = metrics['roc_auc'] >= deploy_threshold
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'detailed_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create a deployment decision file
    deployment_decision = {
        'deploy': deploy,
        'reason': f"Model {'meets' if deploy else 'does not meet'} deployment