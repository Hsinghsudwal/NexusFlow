# MLOps Churn Prediction Pipeline with LocalStack
# ----------------------------------------------
# This is an end-to-end MLOps pipeline for churn prediction that uses
# LocalStack to simulate AWS cloud services locally.

# Required packages:
# pip install boto3 pandas scikit-learn numpy localstack awscli-local sagemaker-python-sdk

import os
import json
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables for LocalStack
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
LOCALSTACK_ENDPOINT = 'http://localhost:4566'

# Step 1: Create necessary AWS resources using LocalStack
def setup_aws_resources():
    """Create all necessary AWS resources in LocalStack"""
    logger.info("Setting up AWS resources in LocalStack")
    
    # Initialize clients
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    sqs = boto3.client('sqs', endpoint_url=LOCALSTACK_ENDPOINT)
    lambda_client = boto3.client('lambda', endpoint_url=LOCALSTACK_ENDPOINT)
    events = boto3.client('events', endpoint_url=LOCALSTACK_ENDPOINT)
    
    # Create S3 buckets
    try:
        s3.create_bucket(Bucket='churn-data-bucket')
        s3.create_bucket(Bucket='churn-models-bucket')
        s3.create_bucket(Bucket='churn-predictions-bucket')
        logger.info("Created S3 buckets")
    except Exception as e:
        logger.warning(f"S3 buckets might already exist: {e}")
    
    # Create SQS queue for model training notifications
    try:
        sqs.create_queue(QueueName='model-training-queue')
        logger.info("Created SQS queue")
    except Exception as e:
        logger.warning(f"SQS queue might already exist: {e}")
    
    # Create EventBridge rule for scheduled retraining
    try:
        # Create a scheduled rule that runs every day
        events.put_rule(
            Name='churn-model-retraining-schedule',
            ScheduleExpression='rate(1 day)',
            State='ENABLED'
        )
        logger.info("Created EventBridge rule")
    except Exception as e:
        logger.warning(f"EventBridge rule might already exist: {e}")
    
    logger.info("AWS resources setup complete")

# Step 2: Data Generation or Loading
def generate_sample_data(n_samples=1000):
    """Generate synthetic churn data for demonstration"""
    logger.info(f"Generating {n_samples} synthetic data samples")
    
    np.random.seed(42)
    
    # Customer features
    tenure = np.random.randint(1, 73, n_samples)
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = monthly_charges * tenure * np.random.uniform(0.9, 1.1, n_samples)
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
    online_security = np.random.choice(['Yes', 'No'], n_samples)
    tech_support = np.random.choice(['Yes', 'No'], n_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    
    # Convert categorical variables to one-hot encoding
    contract_month = (contract_type == 'Month-to-month').astype(int)
    contract_year = (contract_type == 'One year').astype(int)
    contract_two_year = (contract_type == 'Two year').astype(int)
    
    online_sec_yes = (online_security == 'Yes').astype(int)
    tech_support_yes = (tech_support == 'Yes').astype(int)
    
    payment_electronic = (payment_method == 'Electronic check').astype(int)
    payment_mail = (payment_method == 'Mailed check').astype(int)
    payment_bank = (payment_method == 'Bank transfer').astype(int)
    payment_cc = (payment_method == 'Credit card').astype(int)
    
    # Generate churn based on features (simple rule-based generation)
    churn_probability = (
        0.3 - 0.004 * tenure + 
        0.002 * monthly_charges - 
        0.3 * contract_year - 
        0.5 * contract_two_year + 
        0.2 * (1 - online_sec_yes) +
        0.2 * (1 - tech_support_yes) +
        0.1 * payment_electronic
    )
    
    # Clip probabilities to [0, 1] range
    churn_probability = np.clip(churn_probability, 0, 1)
    churn = (np.random.random(n_samples) < churn_probability).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_Month': contract_month,
        'Contract_Year': contract_year,
        'Contract_TwoYear': contract_two_year,
        'OnlineSecurity': online_sec_yes,
        'TechSupport': tech_support_yes,
        'Payment_Electronic': payment_electronic,
        'Payment_Mail': payment_mail,
        'Payment_Bank': payment_bank,
        'Payment_CC': payment_cc,
        'Churn': churn
    })
    
    return data

# Step 3: Data Ingestion and Storage
def upload_data_to_s3(data, bucket_name='churn-data-bucket', key='raw/churn_data.csv'):
    """Upload data to S3 bucket"""
    logger.info(f"Uploading data to S3 bucket {bucket_name}")
    
    # Write DataFrame to CSV
    data.to_csv('/tmp/churn_data.csv', index=False)
    
    # Upload to S3
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    s3.upload_file('/tmp/churn_data.csv', bucket_name, key)
    
    logger.info(f"Data uploaded to s3://{bucket_name}/{key}")
    return f"s3://{bucket_name}/{key}"

# Step 4: Data Processing Pipeline
def process_data(data_location):
    """Process data for model training"""
    logger.info(f"Processing data from {data_location}")
    
    # Parse S3 location
    bucket = data_location.split('/')[2]
    key = '/'.join(data_location.split('/')[3:])
    
    # Download data from S3
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    s3.download_file(bucket, key, '/tmp/raw_data.csv')
    
    # Load data
    data = pd.read_csv('/tmp/raw_data.csv')
    
    # Basic preprocessing
    # Handle missing values if any
    data = data.fillna(0)
    
    # Feature scaling
    features = data.drop('Churn', axis=1)
    target = data['Churn']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Save scaler for inference
    joblib.dump(scaler, '/tmp/scaler.pkl')
    
    # Upload scaler to S3
    s3.upload_file('/tmp/scaler.pkl', 'churn-models-bucket', 'preprocessing/scaler.pkl')
    
    # Prepare processed data
    processed_data = pd.DataFrame(features_scaled, columns=features.columns)
    processed_data['Churn'] = target
    
    # Save processed data
    processed_data.to_csv('/tmp/processed_data.csv', index=False)
    
    # Upload processed data to S3
    processed_data_key = 'processed/churn_data_processed.csv'
    s3.upload_file('/tmp/processed_data.csv', 'churn-data-bucket', processed_data_key)
    
    logger.info(f"Data processing complete. Processed data saved to S3")
    return f"s3://churn-data-bucket/{processed_data_key}"

# Step 5: Model Training
def train_model(processed_data_location):
    """Train a churn prediction model"""
    logger.info(f"Training model using data from {processed_data_location}")
    
    # Parse S3 location
    bucket = processed_data_location.split('/')[2]
    key = '/'.join(processed_data_location.split('/')[3:])
    
    # Download processed data
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    s3.download_file(bucket, key, '/tmp/processed_data.csv')
    
    # Load data
    data = pd.read_csv('/tmp/processed_data.csv')
    
    # Split features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model (Random Forest as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create model metadata
    model_metadata = {
        'model_type': 'RandomForestClassifier',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'data_location': processed_data_location,
        'feature_importance': {name: float(importance) for name, importance in 
                               zip(X.columns, model.feature_importances_)}
    }
    
    # Save model and metadata
    timestamp = int(time.time())
    model_path = f'/tmp/churn_model_{timestamp}.pkl'
    metadata_path = f'/tmp/churn_model_{timestamp}_metadata.json'
    
    joblib.dump(model, model_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f)
    
    # Upload model and metadata to S3
    model_key = f'models/churn_model_{timestamp}.pkl'
    metadata_key = f'models/churn_model_{timestamp}_metadata.json'
    
    s3.upload_file(model_path, 'churn-models-bucket', model_key)
    s3.upload_file(metadata_path, 'churn-models-bucket', metadata_key)
    
    logger.info(f"Model training complete. Model saved to S3 with metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return f"s3://churn-models-bucket/{model_key}", f"s3://churn-models-bucket/{metadata_key}"

# Step 6: Model Registry
def register_model(model_location, metadata_location):
    """Register model in a simple model registry"""
    logger.info(f"Registering model from {model_location} in model registry")
    
    # Parse S3 locations
    model_bucket = model_location.split('/')[2]
    model_key = '/'.join(model_location.split('/')[3:])
    
    metadata_bucket = metadata_location.split('/')[2]
    metadata_key = '/'.join(metadata_location.split('/')[3:])
    
    # Download metadata
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    s3.download_file(metadata_bucket, metadata_key, '/tmp/model_metadata.json')
    
    with open('/tmp/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Get current registry if it exists
    try:
        s3.download_file('churn-models-bucket', 'registry/model_registry.json', '/tmp/model_registry.json')
        with open('/tmp/model_registry.json', 'r') as f:
            registry = json.load(f)
    except:
        registry = {'models': []}
    
    # Add new model to registry
    model_id = model_key.split('/')[-1].replace('.pkl', '')
    model_entry = {
        'model_id': model_id,
        'created_at': metadata['training_date'],
        's3_location': model_location,
        'metadata_location': metadata_location,
        'metrics': metadata['metrics'],
        'status': 'staging'  # Start in staging
    }
    
    # Check if this is better than production model
    current_prod = None
    for model in registry['models']:
        if model['status'] == 'production':
            current_prod = model
            break
    
    # If no production model exists or new model is better, promote to production
    if not current_prod or metadata['metrics']['f1_score'] > current_prod['metrics']['f1_score']:
        if current_prod:
            # Demote current production model
            for model in registry['models']:
                if model['status'] == 'production':
                    model['status'] = 'archived'
                    break
        
        model_entry['status'] = 'production'
        logger.info(f"Model {model_id} promoted to production with F1 score: {metadata['metrics']['f1_score']:.4f}")
    
    # Add to registry
    registry['models'].append(model_entry)
    
    # Save updated registry
    with open('/tmp/model_registry.json', 'w') as f:
        json.dump(registry, f)
    
    # Upload registry to S3
    s3.upload_file('/tmp/model_registry.json', 'churn-models-bucket', 'registry/model_registry.json')
    
    logger.info(f"Model registered with ID: {model_id}")
    return model_id

# Step 7: Deployment
def deploy_model(model_id):
    """Simulate deploying the model for inference"""
    logger.info(f"Deploying model {model_id} for inference")
    
    # Download model registry
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    s3.download_file('churn-models-bucket', 'registry/model_registry.json', '/tmp/model_registry.json')
    
    with open('/tmp/model_registry.json', 'r') as f:
        registry = json.load(f)
    
    # Find model in registry
    model_info = None
    for model in registry['models']:
        if model['model_id'] == model_id:
            model_info = model
            break
    
    if not model_info:
        raise ValueError(f"Model {model_id} not found in registry")
    
    # Get model location
    model_bucket = model_info['s3_location'].split('/')[2]
    model_key = '/'.join(model_info['s3_location'].split('/')[3:])
    
    # Create deployment metadata
    deployment = {
        'model_id': model_id,
        'deployed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'active',
        'endpoint': f"churn-prediction-endpoint-{model_id}"
    }
    
    # Save deployment metadata
    with open('/tmp/deployment.json', 'w') as f:
        json.dump(deployment, f)
    
    # Upload deployment info to S3
    s3.upload_file('/tmp/deployment.json', 'churn-models-bucket', f'deployments/{model_id}.json')
    
    logger.info(f"Model {model_id} deployed successfully to endpoint: {deployment['endpoint']}")
    return deployment['endpoint']

# Step 8: Prediction Service
def prediction_service(endpoint, data):
    """Simulate an inference service using the deployed model"""
    logger.info(f"Making predictions using endpoint: {endpoint}")
    
    # Extract model_id from endpoint
    model_id = endpoint.split('-')[-1]
    
    # Get deployment info
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    s3.download_file('churn-models-bucket', f'deployments/{model_id}.json', '/tmp/deployment.json')
    
    with open('/tmp/deployment.json', 'r') as f:
        deployment = json.load(f)
    
    if deployment['status'] != 'active':
        raise ValueError(f"Endpoint {endpoint} is not active")
    
    # Download registry to get model location
    s3.download_file('churn-models-bucket', 'registry/model_registry.json', '/tmp/model_registry.json')
    
    with open('/tmp/model_registry.json', 'r') as f:
        registry = json.load(f)
    
    # Find model info
    model_info = None
    for model in registry['models']:
        if model['model_id'] == model_id:
            model_info = model
            break
    
    if not model_info:
        raise ValueError(f"Model {model_id} not found in registry")
    
    # Download model
    model_bucket = model_info['s3_location'].split('/')[2]
    model_key = '/'.join(model_info['s3_location'].split('/')[3:])
    
    s3.download_file(model_bucket, model_key, '/tmp/model.pkl')
    
    # Download scaler
    s3.download_file('churn-models-bucket', 'preprocessing/scaler.pkl', '/tmp/scaler.pkl')
    
    # Load model and scaler
    model = joblib.load('/tmp/model.pkl')
    scaler = joblib.load('/tmp/scaler.pkl')
    
    # Preprocess data
    if isinstance(data, pd.DataFrame):
        features = data.copy()
        if 'Churn' in features.columns:
            features = features.drop('Churn', axis=1)
    else:
        # Assume it's a dictionary
        features = pd.DataFrame([data])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]
    
    # Format results
    results = []
    for i, pred in enumerate(predictions):
        result = {
            'prediction': int(pred),
            'churn_probability': float(probabilities[i]),
            'customer_id': f"customer_{i}" if 'customer_id' not in features.columns else features.iloc[i]['customer_id'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_id': model_id
        }
        results.append(result)
    
    # Save predictions to S3
    timestamp = int(time.time())
    predictions_file = f'/tmp/predictions_{timestamp}.json'
    with open(predictions_file, 'w') as f:
        json.dump(results, f)
    
    s3.upload_file(predictions_file, 'churn-predictions-bucket', f'predictions/batch_{timestamp}.json')
    
    logger.info(f"Generated {len(results)} predictions. {sum(predictions)} customers predicted to churn.")
    return results

# Step 9: Monitoring
def setup_monitoring():
    """Set up basic monitoring for the ML pipeline"""
    logger.info("Setting up monitoring for the ML pipeline")
    
    # Create CloudWatch client with LocalStack
    cloudwatch = boto3.client('cloudwatch', endpoint_url=LOCALSTACK_ENDPOINT)
    
    # Create dashboards and alarms
    try:
        # Create dashboard
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["ChurnPrediction", "ModelAccuracy"]
                        ],
                        "period": 86400,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "Model Accuracy"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["ChurnPrediction", "PredictionLatency"]
                        ],
                        "period": 60,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "Prediction Latency"
                    }
                }
            ]
        }
        
        cloudwatch.put_dashboard(
            DashboardName="ChurnPredictionDashboard",
            DashboardBody=json.dumps(dashboard_body)
        )
        
        # Create alarm for model accuracy
        cloudwatch.put_metric_alarm(
            AlarmName="LowModelAccuracyAlarm",
            ComparisonOperator="LessThanThreshold",
            EvaluationPeriods=1,
            MetricName="ModelAccuracy",
            Namespace="ChurnPrediction",
            Period=86400,
            Statistic="Average",
            Threshold=0.7,
            AlarmDescription="Alert when model accuracy drops below 70%",
            DatapointsToAlarm=1
        )
        
        logger.info("Created CloudWatch dashboard and alarms")
    except Exception as e:
        logger.warning(f"Error setting up monitoring: {e}")
    
    return "ChurnPredictionDashboard"

# Step 10: ML Workflow Orchestration
def run_ml_pipeline():
    """Run the full ML pipeline end-to-end"""
    logger.info("Starting full ML pipeline execution")
    
    try:
        # Step 1: Setup AWS resources
        setup_aws_resources()
        
        # Step 2: Generate or load data
        data = generate_sample_data(n_samples=1000)
        
        # Step 3: Upload data to S3
        data_location = upload_data_to_s3(data)
        
        # Step 4: Process data
        processed_data_location = process_data(data_location)
        
        # Step 5: Train model
        model_location, metadata_location = train_model(processed_data_location)
        
        # Step 6: Register model
        model_id = register_model(model_location, metadata_location)
        
        # Step 7: Deploy model
        endpoint = deploy_model(model_id)
        
        # Step 8: Make predictions (sample)
        test_data = data.drop('Churn', axis=1).head(10)
        predictions = prediction_service(endpoint, test_data)
        
        # Step 9: Setup monitoring
        dashboard = setup_monitoring()
        
        # Output summary
        pipeline_summary = {
            "pipeline_execution_id": f"pipeline-{int(time.time())}",
            "execution_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_samples": len(data),
            "model_id": model_id,
            "prediction_endpoint": endpoint,
            "monitoring_dashboard": dashboard,
            "status": "completed"
        }
        
        # Save summary to S3
        with open('/tmp/pipeline_summary.json', 'w') as f:
            json.dump(pipeline_summary, f)
        
        s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
        s3.upload_file('/tmp/pipeline_summary.json', 'churn-models-bucket', f'pipelines/execution_{int(time.time())}.json')
        
        logger.info("ML pipeline executed successfully")
        return pipeline_summary
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

# Step 11: Scheduled Retraining Setup
def setup_scheduled_retraining():
    """Set up scheduled retraining using EventBridge and Lambda (simulated)"""
    logger.info("Setting up scheduled model retraining")
    
    events = boto3.client('events', endpoint_url=LOCALSTACK_ENDPOINT)
    lambda_client = boto3.client('lambda', endpoint_url=LOCALSTACK_ENDPOINT)
    
    # For LocalStack, we'll simulate this by printing the schedule information
    retraining_schedule = {
        "name": "churn-model-retraining",
        "schedule": "rate(1 day)",  # Run daily
        "target_function": "retrain_churn_model",
        "data_source": "s3://churn-data-bucket/raw/",
        "notification_topic": "model-retraining-notifications"
    }
    
    # In a real setup, you would:
    # 1. Create a Lambda function for retraining
    # 2. Set up EventBridge rule to trigger the Lambda function
    # 3. Set up SNS notifications for retraining results
    
    logger.info(f"Scheduled retraining configured to run {retraining_schedule['schedule']}")
    return retraining_schedule

# Main function to run everything
def main():
    """Run the complete MLOps pipeline for churn prediction"""
    logger.info("Starting MLOps Churn Prediction Pipeline")
    
    # Run pipeline
    pipeline_summary = run_ml_pipeline()
    
    # Set up scheduled retraining
    retraining_schedule = setup_scheduled_retraining()
    
    # Print summary
    logger.info("MLOps pipeline setup complete!")
    logger.info(f"Pipeline execution summary: {json.dumps(pipeline_summary, indent=2)}")
    logger.info(f"Retraining schedule: {json.dumps(retraining_schedule, indent=2)}")
    
    return {
        "pipeline": pipeline_summary,
        "retraining": retraining_schedule
    }

if __name__ == "__main__":
    main()
