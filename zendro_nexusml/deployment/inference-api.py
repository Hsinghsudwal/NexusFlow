# FastAPI app for model inference
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import boto3
import joblib
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Connect to LocalStack or AWS based on environment
if ENVIRONMENT == "production":
    # Use actual AWS services
    s3 = boto3.client('s3', region_name=AWS_DEFAULT_REGION)
    sqs = boto3.client('sqs', region_name=AWS_DEFAULT_REGION)
    cloudwatch = boto3.client('cloudwatch', region_name=AWS_DEFAULT_REGION)
else:
    # Use LocalStack
    s3 = boto3.client('s3', endpoint_url=LOCALSTACK_ENDPOINT)
    sqs = boto3.client('sqs', endpoint_url=LOCALSTACK_ENDPOINT)
    cloudwatch = boto3.client('cloudwatch', endpoint_url=LOCALSTACK_ENDPOINT)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and scaler cache
MODEL_CACHE = {}
SCALER_CACHE = None
ACTIVE_MODEL_ID = None

# Pydantic models
class CustomerData(BaseModel):
    tenure: int = Field(..., description="Number of months the customer has been with the company")
    MonthlyCharges: float = Field(..., description="Monthly charges in dollars")
    TotalCharges: float = Field(..., description="Total charges in dollars")
    Contract_Month: int = Field(..., description="Month-to-month contract (1=yes, 0=no)")
    Contract_Year: int = Field(..., description="One year contract (1=yes, 0=no)")
    Contract_TwoYear: int = Field(..., description="Two year contract (1=yes, 0=no)")
    OnlineSecurity: int = Field(..., description="Has online security (1=yes, 0=no)")
    TechSupport: int = Field(..., description="Has tech support (1=yes, 0=no)")
    Payment_Electronic: int = Field(..., description="Uses electronic check (1=yes, 0=no)")
    Payment_Mail: int = Field(..., description="Uses mailed check (1=yes, 0=no)")
    Payment_Bank: int = Field(..., description="Uses bank transfer (1=yes, 0=no)")
    Payment_CC: int = Field(..., description="Uses credit card (1=yes, 0=no)")
    customer_id: Optional[str] = Field(None, description="Customer identifier")

class PredictionRequest(BaseModel):
    customers: List[CustomerData]
    model_id: Optional[str] = Field(None, description="Specific model ID to use (default: active model)")

class PredictionResponse(BaseModel):
    prediction_id: str
    predictions: List[Dict[str, Any]]
    model_id: str
    timestamp: str

class FeedbackData(BaseModel):
    prediction_id: str
    customer_id: str
    actual_churn: bool
    model_id: str
    feedback_notes: Optional[str] = None

class ModelInfo(BaseModel):
    model_id: str
    created_at: str
    metrics: Dict[str, float]
    status: str

# Helper functions
async def load_active_model():
    """Load the active model from the registry"""
    global ACTIVE_MODEL_ID, MODEL_CACHE, SCALER_CACHE
    
    # Download registry
    try:
        s3.download_file('churn-models-bucket', 'registry/model_registry.json', '/tmp/model_registry.json')
        
        with open('/tmp/model_registry.json', 'r') as f:
            registry = json.load(f)
        
        # Find active model
        active_model = None
        for model in registry['models']:
            if model['status'] == 'production':
                active_model = model
                break
        
        if not active_model:
            raise HTTPException(status_code=500, detail="No active model found in registry")
        
        model_id = active_model['model_id']
        
        # Check if model is already loaded
        if model_id == ACTIVE_MODEL_ID and model_id in MODEL_CACHE:
            return model_id
        
        # Load model
        model_bucket = active_model['s3_location'].split('/')[2]
        model_key = '/'.join(active_model['s3_location'].split('/')[3:])
        
        s3.download_file(model_bucket, model_key, '/tmp/model.pkl')
        
        # Load scaler if not already loaded
        if SCALER_CACHE is None:
            s3.download_file('churn-models-bucket', 'preprocessing/scaler.pkl', '/tmp/scaler.pkl')
            SCALER_CACHE = joblib.load('/tmp/scaler.pkl')
        
        # Update cache
        MODEL_CACHE[model_id] = joblib.load('/tmp/model.pkl')
        ACTIVE_MODEL_ID = model_id
        
        logger.info(f"Loaded active model: {model_id}")
        return model_id
        
    except Exception as e:
        logger.error(f"Error loading active model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

async def load_specific_model(model_id):
    """Load a specific model from the registry"""
    global MODEL_CACHE, SCALER_CACHE
    
    # Check if model is already loaded
    if model_id in MODEL_CACHE:
        return model_id
    
    # Download registry
    try:
        s3.download_file('churn-models-bucket', 'registry/model_registry.json', '/tmp/model_registry.json')
        
        with open('/tmp/model_registry.json', 'r') as f:
            registry = json.load(f)
        
        # Find requested model
        model_info = None
        for model in registry['models']:
            if model['model_id'] == model_id:
                model_info = model
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found in registry")
        
        # Load model
        model_bucket = model_info['s3_location'].split('/')[2]
        model_key = '/'.join(model_info['s3_location'].split('/')[3:])
        
        s3.download_file(model_bucket, model_key, '/tmp/model.pkl')
        
        # Load scaler if not already loaded
        if SCALER_CACHE is None:
            s3.download_file('churn-models-bucket', 'preprocessing/scaler.pkl', '/tmp/scaler.pkl')
            SCALER_CACHE = joblib.load('/tmp/scaler.pkl')
        
        # Update cache
        MODEL_CACHE[model_id] = joblib.load('/tmp/model.pkl')
        
        logger.info(f"Loaded specific model: {model_id}")
        return model_id
        
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

async def log_prediction_metrics(background_tasks: BackgroundTasks, prediction_id, model_id, predictions, latency_ms):
    """Log prediction metrics to CloudWatch"""
    
    def _log_metrics():
        try:
            # Log prediction count
            cloudwatch.put_metric_data(
                Namespace="ChurnPrediction",
                MetricData=[
                    {
                        'MetricName': 'PredictionCount',
                        'Value': len(predictions),
                        'Unit': 'Count',
                        'Dimensions': [
                            {
                                'Name': 'ModelId',
                                'Value': model_id
                            }
                        ]
                    }
                ]
            )
            
            # Log latency
            cloudwatch.put_metric_data(
                Namespace="ChurnPrediction",
                MetricData=[
                    {
                        'MetricName': 'PredictionLatency',
                        'Value': latency_ms,
                        'Unit': 'Milliseconds',
                        'Dimensions': [
                            {
                                'Name': 'ModelId',
                                'Value': model_id
                            }
                        ]
                    }
                ]
            )
            
            # Log churn rate
            churn_rate = sum(p['prediction'] for p in predictions) / len(predictions)
            cloudwatch.put_metric_data(
                Namespace="ChurnPrediction",
                MetricData=[
                    {
                        'MetricName': 'ChurnRate',
                        'Value': churn_rate,
                        'Unit': 'None',
                        'Dimensions': [
                            {
                                'Name': 'ModelId',
                                'Value': model_id
                            }
                        ]
                    }
                ]
            )
            
            logger.info(f"Logged metrics for prediction {prediction_id}")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    background_tasks.add_task(_log_metrics)

async def store_prediction_results(background_tasks: BackgroundTasks, prediction_id, model_id, predictions):
    """Store prediction results in S3"""
    
    def _store_results():
        try:
            # Prepare results
            results = {
                'prediction_id': prediction_id,
                'model_id': model_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': predictions
            }
            
            # Save to S3
            prediction_file = f'/tmp/prediction_{prediction_id}.json'
            with open(prediction_file, 'w') as f:
                json.dump(results, f)
            
            s3.upload_file(prediction_file, 'churn-predictions-bucket', f'predictions/{prediction_id}.json')
            logger.info(f"Stored prediction results for {prediction_id}")
            
        except Exception as e:
            logger.error(f"Error storing prediction results: {e}")
    
    background_tasks.add_task(_store_results)

# API routes
@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    start_time = datetime.now()
    
    # Load appropriate model
    if request.model_id:
        model_id = await load_specific_model(request.model_id)
    else:
        model_id = await load_active_model()
    
    # Get model and scaler
    model = MODEL_CACHE[model_id]
    scaler = SCALER_CACHE
    
    # Convert input data to DataFrame
    customers_data = [dict(customer) for customer in request.customers]
    df = pd.DataFrame(customers_data)
    
    # Extract customer IDs, assign UUIDs if not provided
    customer_ids = []
    for i, customer in enumerate(customers_data):
        if customer.get('customer_id'):
            customer_ids.append(customer['customer_id'])
        else:
            customer_ids.append(f"customer_{uuid.uuid4().hex[:8]}")
    
    # Remove customer_id column if it exists
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    # Scale features
    features_scaled = scaler.transform(df)
    
    # Generate predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]
    
    # Format results
    prediction_id = f"pred_{uuid.uuid4().hex}"
    results = []
    
    for i, pred in enumerate(predictions):
        result = {
            'customer_id': customer_ids[i],
            'prediction': int(pred),
            'churn_probability': float(probabilities[i]),
            'churn_risk_level': 'High' if probabilities[i] > 0.7 else 
                                'Medium' if probabilities[i] > 0.4 else 'Low'
        }
        results.append(result)
    
    # Calculate latency
    end_time = datetime.now()
    latency_ms = (end_time - start_time).total_seconds() * 1000
    
    # Log metrics and store results in background
    await log_prediction_metrics(background_tasks, prediction_id, model_id, results, latency_ms)
    await store_prediction_results(background_tasks, prediction_id, model_id, results)
    
    # Return response
    return {
        "prediction_id": prediction_id,
        "predictions": results,
        "model_id": model_id,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackData, background_tasks: BackgroundTasks):
    """Submit feedback on predictions for model evaluation"""
    
    def _store_feedback():
        try:
            # Create feedback record
            feedback_record = dict(feedback)
            feedback_record['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            feedback_record['feedback_id'] = f"feedback_{uuid.uuid4().hex}"
            
            # Save to S3
            feedback_file = f'/tmp/feedback_{feedback_record["feedback_id"]}.json'
            with open(feedback_file, 'w') as f:
                json.dump(feedback_record, f)
            
            s3.upload_file(feedback_file, 'churn-predictions-bucket', f'feedback/{feedback_record["feedback_id"]}.json')
            
            # Send message to SQS for further processing
            sqs.send_message(
                QueueUrl='http://localhost:4566/000000000000/model-training-queue',
                MessageBody=json.dumps({
                    'type': 'feedback',
                    'feedback_id': feedback_record['feedback_id']
                })
            )
            
            logger.info(f"Stored feedback {feedback_record['feedback_id']}")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    background_tasks.add_task(_store_feedback)
    
    return {"status": "success", "message": "Feedback submitted successfully"}

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models from the registry"""
    try:
        # Download registry
        s3.download_file('churn-models-bucket', 'registry/model_registry.json', '/tmp/model_registry.json')
        
        with open('/tmp/model_registry.json', 'r') as f:
            registry = json.load(f)
        
        # Extract model info
        models = []
        for model in registry['models']:
            models.append({
                'model_id': model['model_id'],
                'created_at': model['created_at'],
                'metrics': model['metrics'],
                'status': model['status']
            })
        
        return models
        
    except Exception as e:
        logger.