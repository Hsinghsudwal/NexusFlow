# app/main.py - FastAPI service for model serving

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import os
import time
import mlflow
import redis
import json
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "production-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Connect to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Connect to Redis for feature store and caching
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# Set up metrics
PREDICTION_COUNT = Counter("prediction_count", "Number of predictions made", ["model_version", "status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds", ["model_version"])
FEATURE_DRIFT = Counter("feature_drift_count", "Number of detected feature drifts", ["feature_name"])

app = FastAPI(title="ML Model API", description="API for serving machine learning predictions")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schemas
class PredictionFeatures(BaseModel):
    features: Dict[str, Any]
    
    @validator("features")
    def validate_features(cls, v):
        # This would typically validate required features and types
        if not v:
            raise ValueError("Features cannot be empty")
        return v

class BatchPredictionFeatures(BaseModel):
    instances: List[Dict[str, Any]]
    
    @validator("instances")
    def validate_instances(cls, v):
        if not v:
            raise ValueError("Batch prediction requires at least one instance")
        return v

# Output schemas
class PredictionResponse(BaseModel):
    prediction: Any
    prediction_probability: Optional[Dict[str, float]]
    model_version: str
    prediction_id: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[Any]
    model_version: str
    processing_time_ms: float

# Model loading and caching
def get_model():
    """Load model from MLflow model registry with caching."""
    model_cache_key = f"{MODEL_NAME}:{MODEL_STAGE}"
    cached_model_path = redis_client.get(model_cache_key)
    
    if cached_model_path:
        model_path = cached_model_path.decode("utf-8")
    else:
        try:
            model_path = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            # Cache model path
            redis_client.set(model_cache_key, model_path, ex=3600)  # Expire after 1 hour
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        model = mlflow.pyfunc.load_model(model_path)
        return model, model_path
    except Exception as e:
        logger.error(f"Error loading model from path {model_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Feature processing
def process_features(features: Dict[str, Any]) -> pd.DataFrame:
    """Process and validate input features."""
    try:
        # Convert to DataFrame for prediction
        df = pd.DataFrame([features])
        
        # This would typically include:
        # 1. Feature validation
        # 2. Feature transformation
        # 3. Feature drift detection
        
        # Example drift detection
        for feature, value in features.items():
            # Check if numerical feature is outside expected range
            if isinstance(value, (int, float)):
                feature_key = f"feature_stats:{feature}"
                if redis_client.exists(feature_key):
                    stats = json.loads(redis_client.get(feature_key))
                    if value < stats["min"] or value > stats["max"]:
                        FEATURE_DRIFT.labels(feature_name=feature).inc()
                        logger.warning(f"Detected drift in feature {feature}: {value}")
        
        return df
    except Exception as e:
        logger.error(f"Error processing features: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid features: {str(e)}")

# Background tasks
def log_prediction(prediction_id: str, features: Dict, prediction: Any):
    """Log prediction to storage for monitoring and retraining."""
    try:
        log_entry = {
            "prediction_id": prediction_id,
            "timestamp": time.time(),
            "features": features,
            "prediction": prediction
        }
        redis_client.lpush("prediction_logs", json.dumps(log_entry))
        redis_client.ltrim("prediction_logs", 0, 9999)  # Keep last 10K predictions
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

# API endpoints
@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: PredictionFeatures, 
    background_tasks: BackgroundTasks,
    model_and_path=Depends(get_model)
):
    """Make a single prediction."""
    start_time = time.time()
    model, model_path = model_and_path
    model_version = model_path.split("/")[-1]
    
    try:
        # Process features
        df = process_features(features.features)
        
        # Make prediction
        raw_prediction = model.predict(df)
        prediction = raw_prediction[0]  # First row
        
        # Get prediction probabilities if available
        prediction_probability = None
        try:
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(df)[0]
                classes = model.classes_
                prediction_probability = {str(c): float(p) for c, p in zip(classes, probas)}
        except:
            pass
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time() * 1000)}"
        
        # Log prediction asynchronously
        background_tasks.add_task(
            log_prediction, 
            prediction_id=prediction_id,
            features=features.features,
            prediction=prediction
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.labels(model_version=model_version, status="success").inc()
        PREDICTION_LATENCY.labels(model_version=model_version).observe(processing_time / 1000)
        
        return {
            "prediction": prediction,
            "prediction_probability": prediction_probability,
            "model_version": model_version,
            "prediction_id": prediction_id,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        PREDICTION_COUNT.labels(model_version=model_version, status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(
    batch_features: BatchPredictionFeatures,
    model_and_path=Depends(get_model)
):
    """Make batch predictions."""
    start_time = time.time()
    model, model_path = model_and_path
    model_version = model_path.split("/")[-1]
    
    try:
        # Process all instances
        dfs = []
        for instance in batch_features.instances:
            df = process_features(instance)
            dfs.append(df)
        
        # Combine into a single DataFrame
        batch_df = pd.concat(dfs, ignore_index=True)
        
        # Make predictions
        predictions = model.predict(batch_df).tolist()
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.labels(model_version=model_version, status="success").inc(len(predictions))
        PREDICTION_LATENCY.labels(model_version=model_version).observe(processing_time / 1000)
        
        return {
            "predictions": predictions,
            "model_version": model_version,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        PREDICTION_COUNT.labels(model_version=model_version, status="error").inc(len(batch_features.instances))
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
def model_info(model_and_path=Depends(get_model)):
    """Get information about the current model."""
    _, model_path = model_and_path
    
    try:
        # Get model metadata from MLflow
        client = mlflow.tracking.MlflowClient()
        model_details = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
        
        return {
            "name": MODEL_NAME,
            "version": model_details.version,
            "stage": MODEL_STAGE,
            "creation_timestamp": model_details.creation_timestamp,
            "last_updated_timestamp": model_details.last_updated_timestamp,
            "description": model_details.description,
            "run_id": model_details.run_id
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
