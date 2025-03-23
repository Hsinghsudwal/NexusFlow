from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import time
from datetime import datetime
import json
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
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

# Prometheus metrics
PREDICTION_COUNT = Counter('churn_prediction_count', 'Number of churn predictions made', ['result'])
PREDICTION_LATENCY = Histogram('churn_prediction_latency_seconds', 'Prediction latency in seconds')
ERROR_COUNT = Counter('churn_api_error_count', 'Number of API errors', ['endpoint', 'error_type'])

# Load model from MLflow model registry
def get_model():
    """Load the latest model from MLflow model registry"""
    model_uri = os.getenv("MODEL_URI", "models:/churn-prediction/Production")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model from {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        ERROR_COUNT.labels(endpoint='get_model', error_type='model_loading').inc()
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Initialize model on startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = get_model()

# Input data model
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    internet_service: str
    online_security: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    payment_method: str

# Batch prediction input
class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

# Prediction response
class ChurnPrediction(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    churn_prediction: bool
    prediction_timestamp: str

# Batch prediction response
class BatchPredictionResponse(BaseModel):
    predictions: List[ChurnPrediction]
    model_version: str
    prediction_timestamp: str

# Feature engineering function
def engineer_features(customer: Dict[str, Any]) -> pd.DataFrame:
    """Convert customer data to model features"""
    # Create a dataframe from customer data
    df = pd.DataFrame([customer])
    
    # One-hot encode categorical variables
    cat_cols = ['contract', 'internet_service', 'online_security', 
                'tech_support', 'streaming_tv', 'streaming_movies', 'payment_method']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Ensure all expected columns are present
    expected_columns = joblib.load(os.getenv("COLUMNS_PATH", "/app/models/expected_columns.joblib"))
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Ensure columns are in the right order
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded

@app.post("/predict", response_model=ChurnPrediction)
async def predict(customer: CustomerData, request: Request):
    """Predict churn probability for a single customer"""
    start_time = time.time()
    try:
        # Convert customer data to features
        features = engineer_features(customer.dict())
        
        # Make prediction
        churn_probability = model.predict_proba(features)[0, 1]
        churn_prediction = bool(churn_probability >= 0.5)
        
        # Create response
        response = ChurnPrediction(
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            prediction_timestamp=datetime.now().isoformat()
        )
        
        # Record metrics
        PREDICTION_COUNT.labels(result="churn" if churn_prediction else "no_churn").inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        # Log prediction
        logger.info(f"Prediction: {churn_prediction}, Probability: {churn_probability:.4f}")
        
        return response
    except Exception as e:
        ERROR_COUNT.labels(endpoint='predict', error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(data: BatchCustomerData):
    """Predict churn probability for multiple customers"""
    start_time = time.time()
    try:
        predictions = []
        
        # Process each customer
        for i, customer in enumerate(data.customers):
            features = engineer_features(customer.dict())
            churn_probability = model.predict_proba(features)[0, 1]
            churn_prediction = bool(churn_probability >= 0.5)
            
            predictions.append(ChurnPrediction(
                customer_id=f"customer_{i}" if not hasattr(customer, 'customer_id') else customer.customer_id,
                churn_probability=float(churn_probability),
                churn_prediction=churn_prediction,
                prediction_timestamp=datetime.now().isoformat()
            ))
            
            # Record metrics
            PREDICTION_COUNT.labels(result="churn" if churn_prediction else "no_churn").inc()
        
        # Create response
        response = BatchPredictionResponse(
            predictions=predictions,
            model_version=os.getenv("MODEL_VERSION", "unknown"),
            prediction_timestamp=datetime.now().isoformat()
        )
        
        # Record latency
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return response
    except Exception as e:
        ERROR_COUNT.labels(endpoint='batch_predict', error_type=type(e).__name__).inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": os.getenv("MODEL_VERSION", "unknown")}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return generate_latest()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }
