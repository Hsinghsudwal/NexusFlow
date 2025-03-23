# MLOps Pipeline for Customer Churn Prediction

## 1. Project Structure
```
customer-churn-mlops/
├── .github/workflows/      # CI/CD pipeline definitions
├── data/                   # Data processing scripts
├── models/                 # Model training code
├── pipelines/              # Kubeflow pipeline definitions
├── deployment/             # Kubernetes manifests
│   ├── kubernetes/         # K8s configuration
│   └── kubeflow/           # Kubeflow configuration
├── monitoring/             # Monitoring and alerting
├── api/                    # Model serving API
├── docker/                 # Dockerfiles
│   ├── data-science/       # Dev environment
│   ├── training/           # Training container
│   └── serving/            # Serving container
├── docker-compose.yml      # Local development
└── README.md
```

## 2. Data Pipeline - Extract from AWS RDS

# data/extract.py
import pandas as pd
import boto3
import os
from sqlalchemy import create_engine

def get_db_connection():
    """Create a connection to AWS RDS database"""
    
    # Get credentials from environment or AWS Secrets Manager
    db_host = os.environ.get('DB_HOST')
    db_name = os.environ.get('DB_NAME')
    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    
    # Create SQLAlchemy engine
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    engine = create_engine(connection_string)
    
    return engine

def extract_customer_data():
    """Extract customer data from RDS for churn prediction"""
    
    engine = get_db_connection()
    
    # Example query to extract customer data
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
    FROM customer_data
    WHERE created_at >= NOW() - INTERVAL '3 months'
    """
    
    # Load data into pandas DataFrame
    df = pd.read_sql(query, engine)
    
    # Save to S3 for processing
    save_to_s3(df, 'raw-customer-data.csv')
    
    return df

def save_to_s3(df, filename):
    """Save DataFrame to S3 bucket"""
    
    bucket = os.environ.get('DATA_BUCKET', 'customer-churn-data')
    key = f"raw/{filename}"
    
    # Save locally first
    local_path = f"/tmp/{filename}"
    df.to_csv(local_path, index=False)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, bucket, key)
    
    print(f"Saved data to s3://{bucket}/{key}")

if __name__ == "__main__":
    extract_customer_data()

## 3. Data Preprocessing and Feature Engineering

# data/preprocess.py
import pandas as pd
import numpy as np
import boto3
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_data_from_s3():
    """Load raw customer data from S3"""
    
    bucket = os.environ.get('DATA_BUCKET', 'customer-churn-data')
    key = 'raw/raw-customer-data.csv'
    
    # Download file
    s3_client = boto3.client('s3')
    local_path = '/tmp/raw-customer-data.csv'
    s3_client.download_file(bucket, key, local_path)
    
    # Load into DataFrame
    df = pd.read_csv(local_path)
    return df

def preprocess_data(df):
    """Preprocess customer data for model training"""
    
    # Split features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Define numeric and categorical features
    numeric_features = ['tenure', 'monthly_charges', 'total_charges']
    categorical_features = [
        'contract_type', 'internet_service', 'online_security', 
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_method'
    ]
    
    # Drop customer_id as it's not a feature
    X = X.drop('customer_id', axis=1)
    
    # Define preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Save the preprocessor
    joblib.dump(preprocessor, '/tmp/preprocessor.joblib')
    
    # Upload preprocessor to S3
    s3_client = boto3.client('s3')
    bucket = os.environ.get('MODEL_BUCKET', 'customer-churn-models')
    s3_client.upload_file('/tmp/preprocessor.joblib', bucket, 'preprocessors/latest/preprocessor.joblib')
    
    return X_processed, y, preprocessor

if __name__ == "__main__":
    df = load_data_from_s3()
    X_processed, y, preprocessor = preprocess_data(df)
    
    # Save processed data
    np.save('/tmp/X_processed.npy', X_processed)
    np.save('/tmp/y.npy', y.values)
    
    # Upload processed data to S3
    s3_client = boto3.client('s3')
    bucket = os.environ.get('DATA_BUCKET', 'customer-churn-data')
    s3_client.upload_file('/tmp/X_processed.npy', bucket, 'processed/X_processed.npy')
    s3_client.upload_file('/tmp/y.npy', bucket, 'processed/y.npy')

## 4. Model Training

# models/train.py
import numpy as np
import os
import boto3
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def load_processed_data():
    """Load processed data from S3"""
    
    bucket = os.environ.get('DATA_BUCKET', 'customer-churn-data')
    s3_client = boto3.client('s3')
    
    # Download X and y
    s3_client.download_file(bucket, 'processed/X_processed.npy', '/tmp/X_processed.npy')
    s3_client.download_file(bucket, 'processed/y.npy', '/tmp/y.npy')
    
    # Load arrays
    X = np.load('/tmp/X_processed.npy')
    y = np.load('/tmp/y.npy')
    
    return X, y

def train_model(X, y):
    """Train a Random Forest classifier for churn prediction"""
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob)
    }
    
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model, metrics

def save_model(model, metrics):
    """Save trained model and metrics to S3"""
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model locally
    model_path = f"/tmp/churn_model_{timestamp}.joblib"
    joblib.dump(model, model_path)
    
    # Save metrics locally
    metrics_path = f"/tmp/metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    bucket = os.environ.get('MODEL_BUCKET', 'customer-churn-models')
    
    # Upload model
    model_key = f"models/{timestamp}/model.joblib"
    s3_client.upload_file(model_path, bucket, model_key)
    
    # Upload latest version (for serving)
    s3_client.upload_file(model_path, bucket, "models/latest/model.joblib")
    
    # Upload metrics
    metrics_key = f"models/{timestamp}/metrics.json"
    s3_client.upload_file(metrics_path, bucket, metrics_key)
    
    print(f"Model saved to s3://{bucket}/{model_key}")
    print(f"Metrics saved to s3://{bucket}/{metrics_key}")
    
    return {
        'model_uri': f"s3://{bucket}/{model_key}",
        'metrics_uri': f"s3://{bucket}/{metrics_key}",
        'timestamp': timestamp
    }

if __name__ == "__main__":
    X, y = load_processed_data()
    model, metrics = train_model(X, y)
    save_model(model, metrics)

## 5. Kubeflow Pipeline Definition

# pipelines/churn_pipeline.py
import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath

# Define component for data extraction
def extract_data_op():
    return kfp.components.load_component_from_file('components/extract_data/component.yaml')

# Define component for data preprocessing
def preprocess_data_op():
    return kfp.components.load_component_from_file('components/preprocess_data/component.yaml')

# Define component for model training
def train_model_op():
    return kfp.components.load_component_from_file('components/train_model/component.yaml')

# Define component for model evaluation
def evaluate_model_op():
    return kfp.components.load_component_from_file('components/evaluate_model/component.yaml')

# Define component for model deployment
def deploy_model_op():
    return kfp.components.load_component_from_file('components/deploy_model/component.yaml')

# Define the pipeline
@dsl.pipeline(
    name='Customer Churn Prediction Pipeline',
    description='End-to-end ML pipeline for customer churn prediction'
)
def churn_pipeline():
    # Extract data
    extract_data_task = extract_data_op()
    
    # Preprocess data
    preprocess_task = preprocess_data_op().after(extract_data_task)
    
    # Train model
    train_task = train_model_op().after(preprocess_task)
    
    # Evaluate model
    evaluate_task = evaluate_model_op().after(train_task)
    
    # Deploy model if metrics are good
    with dsl.Condition(evaluate_task.outputs['deploy'] == 'true'):
        deploy_model_op().after(evaluate_task)

# Example component.yaml for extract_data component
"""
name: Extract Customer Data
description: Extracts customer data from AWS RDS
inputs:
  - {name: db_host, type: String}
  - {name: db_name, type: String}
  - {name: db_user, type: String}
  - {name: db_password, type: String}
outputs:
  - {name: output_data, type: Dataset}
implementation:
  container:
    image: ${YOUR_DOCKER_REGISTRY}/extract-data:latest
    command: [python, /app/extract.py]
    args: [
      --db-host, {inputValue: db_host},
      --db-name, {inputValue: db_name},
      --db-user, {inputValue: db_user},
      --db-password, {inputValue: db_password},
      --output-data, {outputPath: output_data}
    ]
"""

## 6. Kubernetes and Docker Configuration

# deployment/kubernetes/model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-model-api
  labels:
    app: churn-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-model-api
  template:
    metadata:
      labels:
        app: churn-model-api
    spec:
      containers:
      - name: model-api
        image: ${YOUR_DOCKER_REGISTRY}/churn-model-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_BUCKET
          value: "customer-churn-models"
        - name: MODEL_PATH
          value: "models/latest/model.joblib"
        - name: PREPROCESSOR_PATH
          value: "preprocessors/latest/preprocessor.joblib"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 15

# deployment/kubernetes/model-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: churn-model-api
spec:
  selector:
    app: churn-model-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

# docker/serving/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ .

# Run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

## 7. Model Serving API

# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import boto3
import numpy as np
import os
import json
from botocore.exceptions import ClientError

app = FastAPI(title="Customer Churn Prediction API")

# Model cache
model = None
preprocessor = None

class CustomerData(BaseModel):
    """Input data for churn prediction"""
    tenure: int
    contract_type: str
    monthly_charges: float
    total_charges: float
    internet_service: str
    online_security: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    payment_method: str

class ChurnPrediction(BaseModel):
    """Output prediction"""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    model_version: str

@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global model, preprocessor
    
    try:
        # Get S3 client
        s3_client = boto3.client('s3')
        bucket = os.environ.get('MODEL_BUCKET', 'customer-churn-models')
        
        # Download model
        model_path = os.environ.get('MODEL_PATH', 'models/latest/model.joblib')
        local_model_path = '/tmp/model.joblib'
        s3_client.download_file(bucket, model_path, local_model_path)
        
        # Download preprocessor
        preprocessor_path = os.environ.get('PREPROCESSOR_PATH', 'preprocessors/latest/preprocessor.joblib')
        local_preprocessor_path = '/tmp/preprocessor.joblib'
        s3_client.download_file(bucket, preprocessor_path, local_preprocessor_path)
        
        # Load both into memory
        model = joblib.load(local_model_path)
        preprocessor = joblib.load(local_preprocessor_path)
        
        print("Model and preprocessor loaded successfully")
    except ClientError as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model could not be loaded")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    """Predict customer churn probability"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame for preprocessing
        customer_dict = customer.dict()
        
        # Generate a customer ID for response
        customer_id = f"pred-{np.random.randint(10000, 99999)}"
        
        # Preprocess input
        customer_df = pd.DataFrame([customer_dict])
        customer_processed = preprocessor.transform(customer_df)
        
        # Make prediction
        churn_prob = model.predict_proba(customer_processed)[0, 1]
        churn_pred = churn_prob >= 0.5
        
        # Get model version
        model_version = os.environ.get('MODEL_VERSION', 'latest')
        
        return {
            "customer_id": customer_id,
            "churn_probability": float(churn_prob),
            "churn_prediction": bool(churn_pred),
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

## 8. CI/CD Pipeline

# .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly retraining

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
        pip install pytest pytest-cov
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Test with pytest
      run: |
        pytest --cov=./ --cov-report=xml
    
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    - name: Build and push Docker images
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
      run: |
        # Build and push training image
        docker build -t $ECR_REGISTRY/churn-training:${{ github.sha }} -f docker/training/Dockerfile .
        docker push $ECR_REGISTRY/churn-training:${{ github.sha }}
        docker tag $ECR_REGISTRY/churn-training:${{ github.sha }} $ECR_REGISTRY/churn-training:latest
        docker push $ECR_REGISTRY/churn-training:latest
        
        # Build and push serving image
        docker build -t $ECR_REGISTRY/churn-serving:${{ github.sha }} -f docker/serving/Dockerfile .
        docker push $ECR_REGISTRY/churn-serving:${{ github.sha }}
        docker tag $ECR_REGISTRY/churn-serving:${{ github.sha }} $ECR_REGISTRY/churn-serving:latest
        docker push $ECR_REGISTRY/churn-serving:latest
  
  deploy-to-eks:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    - name: Update kubeconfig
      run: aws eks update-kubeconfig --name customer-churn-cluster --region us-east-1
    - name: Deploy to Kubernetes
      run: |
        # Update image tags
        sed -i "s|IMAGE_TAG|${{ github.sha }}|g" deployment/kubernetes/model-deployment.yaml
        
        # Apply K8s manifests
        kubectl apply -f deployment/kubernetes/model-deployment.yaml
        kubectl apply -f deployment/kubernetes/model-service.yaml
        
        # Verify deployment
        kubectl rollout status deployment/churn-model-api

## 9. Monitoring and Alerts

# monitoring/prometheus/model-metrics.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: churn-model-metrics
  labels:
    app: churn-model-api
spec:
  selector:
    matchLabels:
      app: churn-model-api
  endpoints:
  - port: http
    path: /metrics
    interval: 15s

# api/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
PREDICTION_COUNT = Counter(
    'churn_prediction_count', 
    'Number of churn predictions',
    ['result']
)

PREDICTION_LATENCY = Histogram(
    'churn_prediction_latency_seconds',
    'Time taken to predict churn',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

MODEL_DRIFT = Gauge(
    'churn_model_drift',
    'Model drift score based on distribution comparison'
)

# Usage in API
def track_prediction(prediction_result, latency):
    result = "churn" if prediction_result else "no_churn"
    PREDICTION_COUNT.labels(result=result).inc()
    PREDICTION_LATENCY.observe(latency)

def update_model_drift(drift_score):
    MODEL_DRIFT.set(drift_score)

# Modified prediction endpoint with metrics
@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    start_time = time.time()
    
    # Original prediction code
    ...
    
    # Track metrics
    latency = time.time() - start_time
    track_prediction(churn_pred, latency)
    
    return {
        "customer_id": customer_id,
        "churn_probability": float(churn_prob),
        "churn_prediction": bool(churn_pred),
        "model_version": model_version
    }

## 10. Model Drift Detection and Retraining

# monitoring/drift_detection.py
import pandas as pd
import numpy as np
import boto3
import os
import json
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

def load_training_distribution():
    """Load the reference distribution from when model was trained"""
    s3_client = boto3.client('s3')
    bucket = os.environ.get('MODEL_BUCKET', 'customer-churn-models')
    key = 'reference/feature_distribution.json'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        distribution = json.loads(response['Body'].read().decode('utf-8'))
        return distribution
    except Exception as e:
        print(f"Error loading reference distribution: {e}")
        return None

def get_recent_predictions():
    """Get recent predictions from logging database"""
    # This would connect to your prediction logging system
    # For example purposes, we'll simulate some data
    
    query = """
    SELECT 
        input_features,
        prediction,
        actual_label,
        prediction_time
    FROM prediction_logs
    WHERE prediction_time >= NOW() - INTERVAL '7 days'
    """
    
    # This would be real data from your database
    recent_features = np.random.normal(0, 1, (1000, 10))
    recent_predictions = np.random.randint(0, 2, 1000)
    recent_actuals = np.random.randint(0, 2, 1000)
    
    return recent_features, recent_predictions, recent_actuals

def calculate_drift(reference_dist, current_features):
    """Calculate distribution drift using KS test"""
    drift_scores = {}
    
    # For each feature, calculate KS statistic
    for i, feature_name in enumerate(reference_dist.keys()):
        reference_values = reference_dist[feature_name]
        current_values = current_features[:, i]
        
        # Run Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(reference_values, current_values)
        drift_scores[feature_name] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'drift_detected': p_value < 0.05
        }
    
    # Calculate overall drift score (average KS statistic)
    overall_drift = sum(d['ks_statistic'] for d in drift_scores.values()) / len(drift_scores)
    
    return drift_scores, overall_drift

def calculate_performance(predictions, actuals):
    """Calculate model performance metrics"""
    accuracy = accuracy_score(actuals, predictions)
    
    # Calculate other metrics as needed
    
    return {
        'accuracy': accuracy,
        'sample_size': len(predictions)
    }

def trigger_retraining(drift_scores, performance_metrics):
    """Trigger model retraining pipeline if needed"""
    # Define thresholds
    DRIFT_THRESHOLD = 0.2
    ACCURACY_THRESHOLD = 0.75
    
    # Check if we should retrain
    should_retrain = False
    reason = []
    
    # Check drift
    if drift_scores['overall_drift'] > DRIFT_THRESHOLD:
        should_retrain = True
        reason.append(f"Data drift detected: {drift_scores['overall_drift']:.4f} > {DRIFT_THRESHOLD}")
    
    # Check performance
    if performance_metrics['accuracy'] < ACCURACY_THRESHOLD:
        should_retrain = True
        reason.append(f"Performance degradation: {performance_metrics['accuracy']:.4f} < {ACCURACY_THRESHOLD}")
    
    if should_retrain:
        # Log retraining trigger
        print(f"Triggering model retraining due to: {', '.join(reason)}")
        
        # Trigger Kubeflow pipeline
        run_id = trigger_kubeflow_pipeline()
        
        return {
            'retraining_triggered': True,
            'reasons': reason,
            'pipeline_run_id': run_id
        }
    else:
        return {
            'retraining_triggered': False,
            'metrics': {
                'drift': drift_scores['overall_drift'],
                'accuracy': performance_metrics['accuracy']
            }
        }

def trigger_kubeflow_pipeline():
    """Trigger Kubeflow pipeline for model retraining"""
    # Implementation depends on your Kubeflow setup
    # Here's a simplified example
    
    import kfp
    client = kfp.Client()
    
    # Run the pipeline
    run = client.run_pipeline(
        experiment_id='churn-prediction',
        job_name=f'retraining-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        pipeline_id='customer-churn-pipeline'
    )
    
    return run.id

def main():
    # Load reference distribution
    reference_dist = load_training_distribution()
    
    # Get recent predictions
    features, predictions, actuals = get_recent_predictions()
    
    # Calculate drift
    drift_scores, overall_drift = calculate_drift(reference_dist, features)
    drift_scores['overall_drift'] = overall_drift
    
    # Calculate performance
    performance = calculate_performance(predictions, actuals)
    
    # Decide if retraining is needed
    retraining_result = trigger_retraining(drift_scores, performance)
    
    # Log metrics to CloudWatch
    log_metrics(drift_scores, performance)
    
    # Store results
    save_monitoring_results(drift_scores, performance, retraining_result)

if __name__ == "__main__":
    main()

## 11. Docker Compose for Local Development

# docker-compose.yml
version: '3'

services:
  # PostgreSQL database for local development
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: customer_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Jupyter notebook for data science work
  jupyter:
    build:
      context: .
      dockerfile: docker/data-science/Dockerfile
    