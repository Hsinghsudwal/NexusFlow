# MLOps Framework Implementation with Code Examples

# ===============================
# 1. DATA ENGINEERING PIPELINE
# ===============================

# 1.1 Data Collection with Airflow
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd

def fetch_data_from_source():
    """Fetch data from an external source"""
    # Example: Fetch data from API
    import requests
    response = requests.get("https://api.example.com/data")
    data = response.json()
    
    # Convert to DataFrame and save locally
    df = pd.DataFrame(data)
    df.to_csv("/tmp/raw_data.csv", index=False)
    return "/tmp/raw_data.csv"

def upload_to_data_lake(file_path):
    """Upload collected data to S3 data lake"""
    # Add date-based partitioning
    today = datetime.today().strftime('%Y/%m/%d')
    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_file(
        filename=file_path,
        key=f'data_lake/raw/{today}/data.csv',
        bucket_name='mlops-data-lake',
        replace=True
    )

# Define Airflow DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['data-team@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

data_collection_dag = DAG(
    'data_collection_pipeline',
    default_args=default_args,
    description='Collect data from sources and store in data lake',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'data-engineering'],
)

fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_source,
    dag=data_collection_dag,
)

upload_task = PythonOperator(
    task_id='upload_to_lake',
    python_callable=upload_to_data_lake,
    op_kwargs={'file_path': "{{ task_instance.xcom_pull(task_ids='fetch_data') }}"},
    dag=data_collection_dag,
)

fetch_task >> upload_task

# 1.2 Data Validation with Great Expectations
import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.checkpoint import SimpleCheckpoint

def validate_data():
    # Initialize Great Expectations context
    context = ge.get_context()
    
    # Create batch request for the data
    batch_request = BatchRequest(
        datasource_name="my_s3_datasource",
        data_connector_name="default_inferred_data_connector_name",
        data_asset_name="data_lake/raw",
        batch_spec_passthrough={"reader_method": "read_csv", "reader_options": {"sep": ","}}
    )
    
    # Load expectation suite
    expectation_suite_name = "data_quality_suite"
    
    # Create a checkpoint for validation
    checkpoint_config = {
        "name": "data_validation_checkpoint",
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S",
        "validations": [
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name
            }
        ]
    }
    
    checkpoint = SimpleCheckpoint(
        name=checkpoint_config["name"],
        data_context=context,
        config_version=checkpoint_config["config_version"],
        run_name_template=checkpoint_config["run_name_template"],
        validations=checkpoint_config["validations"],
    )
    
    # Run the checkpoint
    results = checkpoint.run()
    
    # Raise exception if validation fails
    if not results["success"]:
        raise Exception("Data validation failed!")
    
    return results

# 1.3 Data Preprocessing with Scikit-learn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

def create_preprocessing_pipeline(data_path):
    """Create and save preprocessing pipeline"""
    # Load the data
    df = pd.read_csv(data_path)
    
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Define preprocessing for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create and save the preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit the pipeline on the data
    preprocessing_pipeline.fit(df)
    
    # Save the pipeline
    with open('preprocessing_pipeline.pkl', 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    
    return 'preprocessing_pipeline.pkl'

# 1.4 Feature Store with Feast
from datetime import timedelta
import pandas as pd
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.repo_config import RepoConfig

# Define entity
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Customer identifier",
)

# Define feature data source
customer_features_source = FileSource(
    path="s3://mlops-data-lake/features/customer_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
    data_format=ParquetFormat(),
)

# Define feature view
customer_features_view = FeatureView(
    name="customer_features",
    entities=["customer_id"],
    ttl=timedelta(days=90),
    features=[
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="gender", dtype=ValueType.STRING),
        Feature(name="total_purchase_amount", dtype=ValueType.FLOAT),
        Feature(name="num_purchases", dtype=ValueType.INT64),
        Feature(name="avg_purchase_amount", dtype=ValueType.FLOAT),
    ],
    online=True,
    input=customer_features_source,
    tags={"team": "customer_analytics"},
)

# Feast feature_store.yaml configuration
feast_config = RepoConfig(
    registry="s3://mlops-feast-registry/registry.db",
    project="feature_store",
    provider="aws",
    online_store={
        "type": "dynamodb",
        "region": "us-west-2",
    },
    offline_store={
        "type": "file",
    },
    entity_key_serialization_version=2,
)

# ===============================
# 2. MODEL DEVELOPMENT PIPELINE
# ===============================

# 2.1 Experiment Tracking with MLflow
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model_with_tracking(data_path, preprocessing_pipeline_path):
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("customer_churn_prediction")
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    
    # Load preprocessing pipeline
    with open(preprocessing_pipeline_path, 'rb') as f:
        preprocessing_pipeline = pickle.load(f)
    
    # Preprocess data
    X_processed = preprocessing_pipeline.transform(X)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log preprocessing pipeline
        mlflow.sklearn.log_model(preprocessing_pipeline, "preprocessing_pipeline")
        
        # Set model parameters
        n_estimators = 100
        max_depth = 10
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_processed, y)
        
        # Make predictions
        y_pred = model.predict(X_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importances.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")
        
        return run.info.run_id

# 2.2 Hyperparameter Optimization with Optuna
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial, X, y):
    # Define hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 4, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    
    # Create model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Evaluate model using cross-validation
    score = cross_val_score(model, X, y, cv=5, scoring='f1')
    
    # Return the mean score
    return score.mean()

def optimize_hyperparameters(X, y, n_trials=100):
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    
    # Log to MLflow
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1", study.best_value)
        
        # Create visualization
        optuna.visualization.plot_param_importances(study).write_image("param_importances.png")
        mlflow.log_artifact("param_importances.png")
        
        # Train model with best parameters
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X, y)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "optimized_model")
        
        return run.info.run_id, best_params

# 2.3 Model Registry and Versioning
def register_model(run_id, model_name="customer_churn_model"):
    # Register model to MLflow Model Registry
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri, model_name)
    
    # Add description
    client = mlflow.tracking.MlflowClient()
    client.update_registered_model(
        name=model_name,
        description="Customer churn prediction model"
    )
    
    # Add model tag
    client.set_registered_model_tag(
        name=model_name,
        key="team",
        value="customer_analytics"
    )
    
    # Transition model to staging
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Staging"
    )
    
    return registered_model.version

# ===============================
# 3. MODEL DEPLOYMENT PIPELINE
# ===============================

# 3.1 Model Packaging with FastAPI and Docker
# app.py for FastAPI application

def create_fastapi_app():
    """Create a FastAPI app file"""
    app_code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
from typing import List

app = FastAPI(title="Churn Prediction API", description="API for customer churn prediction")

# Load model from MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
model_name = "customer_churn_model"
stage = "Production"  # or "Staging", "None"

# Get latest model version from registry
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

class Customer(BaseModel):
    customer_id: int
    age: int
    gender: str
    tenure: int
    usage_frequency: float
    support_calls: int
    payment_delay: int
    subscription_type: str
    
class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    churn_prediction: bool
    
class BatchPredictionRequest(BaseModel):
    customers: List[Customer]

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: Customer):
    # Convert customer data to DataFrame
    customer_dict = customer.dict()
    customer_id = customer_dict.pop("customer_id")
    df = pd.DataFrame([customer_dict])
    
    # Make prediction
    try:
        prediction_result = model.predict(df)
        probability = float(prediction_result[0])
        prediction = bool(probability >= 0.5)
        
        return {
            "customer_id": customer_id,
            "churn_probability": probability,
            "churn_prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict_churn(request: BatchPredictionRequest):
    # Extract customer IDs and create DataFrame
    customer_ids = [customer.customer_id for customer in request.customers]
    customer_data = [customer.dict() for customer in request.customers]
    
    # Remove customer_id from prediction data
    for customer in customer_data:
        customer.pop("customer_id")
    
    df = pd.DataFrame(customer_data)
    
    # Make predictions
    try:
        probabilities = model.predict(df)
        predictions = [bool(p >= 0.5) for p in probabilities]
        
        # Create response
        results = [
            {
                "customer_id": cid,
                "churn_probability": float(prob),
                "churn_prediction": pred
            }
            for cid, prob, pred in zip(customer_ids, probabilities, predictions)
        ]
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    """
    
    with open("app.py", "w") as f:
        f.write(app_code)
    
    return "app.py"

def create_dockerfile():
    """Create a Dockerfile for the FastAPI app"""
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    return "Dockerfile"

def create_requirements_file():
    """Create requirements.txt for the FastAPI app"""
    requirements = """
fastapi==0.88.0
uvicorn==0.20.0
mlflow==2.4.0
pandas==1.5.3
numpy==1.24.0
scikit-learn==1.2.0
pydantic==1.10.4
    """
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    return "requirements.txt"

# 3.2 Kubernetes Deployment
def create_kubernetes_manifests():
    """Create Kubernetes deployment and service manifests"""
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction-api
  namespace: mlops
  labels:
    app: churn-prediction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: churn-prediction-api
    spec:
      containers:
      - name: churn-prediction-api
        image: ${ECR_REPO}/churn-prediction-api:${IMAGE_TAG}
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: LOG_LEVEL
          value: "INFO"
    """
    
    service_yaml = """
apiVersion: v1
kind: Service
metadata:
  name: churn-prediction-api
  namespace: mlops
  labels:
    app: churn-prediction-api
spec:
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: churn-prediction-api
  type: ClusterIP
    """
    
    hpa_yaml = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-prediction-api
  namespace: mlops
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
    """
    
    with open("k8s-deployment.yaml", "w") as f:
        f.write(deployment_yaml)
    
    with open("k8s-service.yaml", "w") as f:
        f.write(service_yaml)
    
    with open("k8s-hpa.yaml", "w") as f:
        f.write(hpa_yaml)
    
    return ["k8s-deployment.yaml", "k8s-service.yaml", "k8s-hpa.yaml"]

# 3.3 Continuous Deployment with GitHub Actions
def create_github_actions_workflow():
    """Create GitHub Actions workflow for CI/CD"""
    workflow_yaml = """
name: MLOps CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
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
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: churn-prediction-api
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"

  deploy-staging:
    needs: build
    if: github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
      - name: Update deployment image
        run: |
          export ECR_REPO="${{ steps.login-ecr.outputs.registry }}/churn-prediction-api"
          export IMAGE_TAG="${{ github.sha }}"
          envsubst < k8s-deployment.yaml > k8s-deployment-updated.yaml
      - name: Deploy to EKS
        uses: Azure/k8s-deploy@v1
        with:
          manifests: |
            k8s-deployment-updated.yaml
            k8s-service.yaml
            k8s-hpa.yaml
          namespace: mlops
          cluster-name: mlops-eks-cluster
          token: ${{ secrets.K8S_TOKEN }}

  promote-to-production:
    needs: deploy-staging
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://api.example.com/churn-prediction
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
      - name: Promote model to production
        run: |
          # Install MLflow
          pip install mlflow
          
          # Set MLflow tracking URI
          export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
          
          # Get latest staging model version
          VERSION=$(mlflow models get-latest-versions --name customer_churn_model --stages Staging | jq '.[0].version')
          
          # Promote to production
          mlflow models transition-stage --name customer_churn_model --version $VERSION --stage Production
      - name: Deploy to production EKS
        uses: Azure/k8s-deploy@v1
        with:
          manifests: |
            k8s-deployment-updated.yaml
            k8s-service.yaml
            k8s-hpa.yaml
          namespace: mlops-prod
          cluster-name: mlops-eks-cluster
          token: ${{ secrets.K8S_PROD_TOKEN }}
    """
    
    with open(".github/workflows/mlops-cicd.yml", "w") as f:
        f.write(workflow_yaml)
    
    return ".github/workflows/mlops-cicd.yml"

# ===============================
# 4. MONITORING PIPELINE
# ===============================

# 4.1 Model Monitoring with Prometheus and Evidently
import json
from datetime import datetime
from typing import Dict, List, Optional

class ModelMonitor:
    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self.metrics = {}
        
    def log_prediction(self, 
                       input_data: Dict, 
                       prediction: float, 
                       actual: Optional[float] = None,
                       metadata: Optional[Dict] = None):
        """Log a single prediction for monitoring"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "version": self.version,
            "input_data": input_data,
            "prediction": prediction
        }
        
        if actual is not None:
            log_entry["actual"] = actual
        
        if metadata is not None:
            log_entry["metadata"] = metadata
            
        # In a real implementation, this would be sent to a monitoring system
        # For example, storing in a database or sending to Kafka
        print(json.dumps(log_entry))
        
        # Update metrics
        self._update_metrics(input_data, prediction, actual)
        
    def _update_metrics(self, input_data: Dict, prediction: float, actual: Optional[float]):
        """Update monitoring metrics"""
        # Record prediction distribution
        self.metrics.setdefault("prediction_count", 0)
        self.metrics["prediction_count"] += 1
        
        # Record prediction value distribution
        self.metrics.setdefault("prediction_sum", 0)
        self.metrics["prediction_sum"] += prediction
        
        # Calculate prediction mean
        self.metrics["prediction_mean"] = self.metrics["prediction_sum"] / self.metrics["prediction_count"]
        
        # If we have ground truth, calculate accuracy metrics
        if actual is not None:
            self.metrics.setdefault("accuracy_metrics", {"count": 0, "error_sum": 0})
            self.metrics["accuracy_metrics"]["count"] += 1
            self.metrics["accuracy_metrics"]["error_sum"] += abs(prediction - actual)
            self.metrics["accuracy_metrics"]["mae"] = (
                self.metrics["accuracy_metrics"]["error_sum"] / 
                self.metrics["accuracy_metrics"]["count"]
            )
    
    def get_metrics(self):
        """Get current monitoring metrics"""
        return self.metrics

# Example usage of the model monitor
def monitor_example():
    # Create monitor
    monitor = ModelMonitor(model_name="customer_churn_model", version="1")
    
    # Log predictions
    monitor.log_prediction(
        input_data={"age": 35, "tenure": 24, "subscription_type": "premium"},
        prediction=0.75
    )
    
    # Log prediction with ground truth
    monitor.log_prediction(
        input_data={"age": 42, "tenure": 6, "subscription_type": "basic"},
        prediction=0.85,
        actual=1.0
    )
    
    # Get metrics
    metrics = monitor.get_metrics()
    print(f"Monitoring metrics: {metrics}")

# 4.2 Drift Detection with Evidently AI
from datetime import datetime
import pandas as pd
import numpy as np

def generate_drift_report(reference_data_path, current_data_path, output_path):
    """Generate drift report using Evidently"""
    # In a real implementation, you would use Evidently here
    # This is a placeholder that captures the core concepts
    
    drift_code = """
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

# Load reference and current data
reference_data = pd.read_csv("reference_data.csv")
current_data = pd.read_csv("current_data.csv")

# Create a data drift report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="tenure"),
    ColumnDriftMetric(column_name="subscription_type")
])

# Calculate drift
data_drift_report.run(reference_data=reference_data, current_data=current_data)

# Save the report
data_drift_report.save_html("data_drift_report.html")

# If labels are available, check for target drift
if "churn" in reference_data.columns and "churn" in current_data.columns:
    target_drift_