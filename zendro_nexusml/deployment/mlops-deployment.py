
# src/deployment/model_deployer.py
import os
import pickle
import boto3
import shutil
from fastapi import FastAPI
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelDeployer:
    """Class for deploying ML models to different environments."""
    
    def __init__(self, config):
        self.config = config
        
        # Set up AWS S3 client (for LocalStack)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.config.AWS_ENDPOINT_URL,
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1'
        )
    
    def deploy_to_production(self, model_path=None):
        """
        Deploy a model to the production environment.
        
        Args:
            model_path: Path to the model file
                        Defaults to the production model path
        
        Returns:
            bool: True if successful
        """
        if model_path is None:
            model_path = self.config.PROD_MODEL_PATH
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Copy model to deployment directory
        deploy_dir = os.path.join(self.config.BASE_DIR, "deployment")
        os.makedirs(deploy_dir, exist_ok=True)
        
        deploy_path = os.path.join(deploy_dir, "production_model.pkl")
        shutil.copy(model_path, deploy_path)
        
        logger.info(f"Model deployed to {deploy_path}")
        
        # Upload to S3
        try:
            self.s3_client.upload_file(
                model_path,
                self.config.S3_BUCKET_NAME,
                "deployed/production_model.pkl"
            )
            logger.info(f"Model uploaded to S3 bucket: {self.config.S3_BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
        
        return True
    
    def create_deployment_metadata(self):
        """Create metadata for the deployed model."""
        import json
        from datetime import datetime
        
        metadata = {
            "model_version": self.config.MODEL_VERSION,
            "deployment_time": datetime.now().isoformat(),
            "model_type": self.config.MODEL_TYPE,
            "description": f"Customer Churn Prediction Model - {self.config.MODEL_TYPE}"
        }
        
        # Save metadata locally
        deploy_dir = os.path.join(self.config.BASE_DIR, "deployment")
        metadata_path = os.path.join(deploy_dir, "model_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Upload to S3
        try:
            self.s3_client.put_object(
                Bucket=self.config.S3_BUCKET_NAME,
                Key="deployed/model_metadata.json",
                Body=json.dumps(metadata)
            )
        except Exception as e:
            logger.error(f"Error uploading metadata to S3: {str(e)}")
        
        return metadata

# src/deployment/api.py

# src/deployment/api.py
import os
import pickle
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Customer Churn Prediction API", 
              description="API for predicting customer churn",
              version="1.0.0")

# Define request and response models
class PredictionRequest(BaseModel):
    features: dict
    request_id: str = None

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    request_id: str
    model_version: str
    timestamp: str

# Global variables to store model and preprocessor
model = None
preprocessor = None
config = None

def load_model(model_path=None):
    """Load the production model."""
    global model, config
    
    if model_path is None and config is not None:
        model_path = config.PROD_MODEL_PATH
    
    if model_path is None:
        raise ValueError("Model path not specified and config not available")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def load_preprocessor(preprocessor_path=None):
    """Load the preprocessing pipeline."""
    global preprocessor, config
    
    if preprocessor_path is None and config is not None:
        # Find latest preprocessor
        model_dir = config.MODELS_DIR
        preproc_files = [f for f in os.listdir(model_dir) if f.startswith("preprocessing_pipeline_")]
        if preproc_files:
            latest_preproc = sorted(preproc_files)[-1]
            preprocessor_path = os.path.join(model_dir, latest_preproc)
    
    if preprocessor_path is None:
        raise ValueError("Preprocessor path not specified and couldn't find a preprocessor")
    
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading preprocessor: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    # Load configuration
    from src.config.pipeline_config import PipelineConfig
    global config
    config = PipelineConfig()
    
    # Load model and preprocessor
    load_model()
    load_preprocessor()

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "status": "active"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None or preprocessor is None:
        return {"status": "error", "message": "Model or preprocessor not loaded"}
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict customer churn based on input features.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    # Prepare request ID
    request_id = request.request_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # Preprocess input data
        X = preprocessor.transform(df)
        
        # Make prediction
        probability = model.predict_proba(X)[0, 1]
        prediction = int(probability >= 0.5)
        
        # Log prediction
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Request ID: {request_id}")
        
        # Store prediction for monitoring
        try:
            prediction_log = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "features": request.features,
                "prediction": prediction,
                "probability": float(probability),
                "model_version": config.MODEL_VERSION
            }
            
            log_dir = os.path.join(config.LOGS_DIR, "predictions")
            os.makedirs(log_dir, exist_ok=True)
            
            with open(os.path.join(log_dir, f"{request_id}.json"), 'w') as f:
                json.dump(prediction_log, f)
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
        
        # Return response
        return PredictionResponse(
            prediction=prediction,
            probability=float(probability),
            request_id=request_id,
            model_version=config.MODEL_VERSION,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def run_api(host=None, port=None):
    """Run the API server."""
    from src.config.pipeline_config import PipelineConfig
    config = PipelineConfig()
    
    host = host or config.API_HOST
    port = port or config.API_PORT
    
    uvicorn.run("src.deployment.api:app", host=host, port=port, reload=True)

# src/monitoring/data_drift_detector.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataDriftDetector:
    """Class for detecting drift in input data over time."""
    
    def __init__(self, config):
        self.config = config
        self.reference_data = None
    
    def load_reference_data(self, data=None, file_path=None):
        """
        Load reference data for drift comparison.
        
        Args:
            data: Pandas DataFrame with reference data
            file_path: Path to saved reference data
        """
        if data is not None:
            self.reference_data = data
            
            # Save reference data for future use
            ref_data_path = os.path.join(
                self.config.MODELS_DIR, 
                f"reference_data_{self.config.MODEL_VERSION}.pkl"
            )
            data.to_pickle(ref_data_path)
            logger.info(f"Reference data saved to {ref_data_path}")
        
        elif file_path is not None and os.path.exists(file_path):
            self.reference_data = pd.read_pickle(file_path)
            logger.info(f"Reference data loaded from {file_path}")
        
        else:
            # Try to find latest reference data
            model_dir = self.config.MODELS_DIR
            ref_files = [f for f in os.listdir(model_dir) if f.startswith("reference_data_")]
            
            if not ref_files:
                logger.error("No reference data found")
                return False
            
            latest_ref = sorted(ref_files)[-1]
            self.reference_data = pd.read_pickle(os.path.join(model_dir, latest_ref))
            logger.info(f"Reference data loaded from {latest_ref}")
        
        return True
    
    def collect_recent_data(self, days=7):
        """
        Collect data from recent predictions.
        
        Args:
            days: Number of days to look back
        
        Returns:
            DataFrame with recent data
        """
        log_dir = os.path.join(self.config.LOGS_DIR, "predictions")
        if not os.path.exists(log_dir):
            logger.error(f"Prediction logs directory not found: {log_dir}")
            return None
        
        # Get log files
        files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
        
        if not files:
            logger.error("No prediction logs found")
            return None
        
        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=days)
        
        # Load and combine prediction data
        data_list = []
        
        for file in files:
            try:
                with open(os.path.join(log_dir, file), 'r') as f:
                    log = json.load(f)
                
                # Check if within time range
                timestamp = datetime.fromisoformat(log["timestamp"])
                if timestamp >= cutoff:
                    data_list.append(log["features"])
            except Exception as e:
                logger.error(f"Error reading log file {file}: {str(e)}")
        
        if not data_list:
            logger.error(f"No prediction data found in the last {days} days")
            return None
        
        return pd.DataFrame(data_list)
    
    def detect_drift(self, current_data=None, threshold=None):
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current data to compare against reference
            threshold: Drift threshold (p-value)
                       Defaults to config value
        
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            logger.error("Reference data not loaded")
            return None
        
        if current_data is None:
            current_data = self.collect_recent_data()
            
            if current_data is None:
                logger.error("Failed to collect recent data")
                return None
        
        if threshold is None:
            threshold = self.config.DRIFT_THRESHOLD
        
        logger.info("Detecting data drift")
        
        # Find common columns
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        
        if not common_cols:
            logger.error("No common columns found between reference and current data")
            return None
        
        # Calculate drift for each feature
        drift_results = {}
        drifted_features = []
        
        for col in common_cols:
            try:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(self.reference_data[col]) or not pd.api.types.is_numeric_dtype(current_data[col]):
                    continue
                
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = ks_2samp(
                    self.reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                
                drift_results[col] = {
                    "statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < threshold
                }
                
                if p_value < threshold:
                    drifted_features.append(col)
                    logger.warning(f"Drift detected in feature '{col}': p-value = {p_value:.6f}")
            except Exception as e:
                logger.error(f"Error calculating drift for feature '{col}': {str(e)}")
        
        # Overall drift result
        drift_detected = len(drifted_features) > 0
        
        result = {
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
            "num_drifted_features": len(drifted_features),
            "total_features": len(common_cols),
            "drift_percentage": len(drifted_features) / len(common_cols) if common_cols else 0,
            "feature_results": drift_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save drift report
        report_path = os.path.join(
            self.config.LOGS_DIR, 
            f"drift_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"Drift report saved to {report_path}")
        
        if drift_detected:
            logger.warning(f"Data drift detected in {len(drifted_features)} features!")
        else:
            logger.info("No significant data drift detected")
        
        return result

# src/monitoring/model_monitor.py
import os
import json
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelMonitor:
    """Class for monitoring model performance and data in production."""
    
    def __init__(self, config, drift_detector=None):
        self.config = config
        self.drift_detector = drift_detector
        self.monitoring_thread = None
        self.stop_monitoring = False
    
    def calculate_performance_metrics(self, days=30):
        """
        Calculate performance metrics from recent predictions with feedback.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dictionary with performance metrics
        """
        log_dir = os.path.join(self.config.LOGS_DIR, "feedback")
        if not os.path.exists(log_dir):
            logger.error(f"Feedback logs directory not found: {log_dir}")
            return None
        
        # Get log files
        files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
        
        if not files:
            logger.error("No feedback logs found")
            return None
        
        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=days)
        
        # Load feedback data
        predictions = []
        actuals = []
        
        for file in files:
            try:
                with open(os.path.join(log_dir, file), 'r') as f:
                    log = json.load(f)
                
                # Check if within time range
                timestamp = datetime.fromisoformat(log["timestamp"])
                if timestamp >= cutoff:
                    predictions.append(log["prediction"])
                    actuals.append(log["actual"])
            except Exception as e:
                logger.error(f"Error reading feedback log {file}: {str(e)}")
        
        if not predictions:
            logger.error(f"No feedback data found in the last {days} days")
            return None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "accuracy": float(accuracy_score(actuals, predictions)),
            "precision": float(precision_score(actuals, predictions, average='weighted')),
            "recall": float(recall_score(actuals, predictions, average='weighted')),
            "f1": float(f1_score(actuals, predictions, average='weighted')),
            "num_samples": len(predictions),
            "time_period": f"last_{days}_days",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Production performance: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        
        return metrics
    
    def check_volume_metrics(self, days=1):
        """
        Check prediction volume metrics.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dictionary with volume metrics
        """
        log_dir = os.path.join(self.config.LOGS_DIR, "predictions")
        if not os.path.exists(log_dir):
            logger.error(f"Prediction logs directory not found: {log_dir}")
            return None
        
        # Get log files
        files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
        
        if not files:
            logger.warning("No prediction logs found")
            return {"total_predictions": 0}
        
        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=days)
        
        # Count predictions and extract timestamps
        timestamps = []
        
        for file in files:
            try:
                with open(os.path.join(log_dir, file), 'r') as f:
                    log = json.load(f)
                
                # Check if within time range
                timestamp = datetime.fromisoformat(log["timestamp"])
                if timestamp >= cutoff:
                    timestamps.append(timestamp)
            except Exception as e:
                logger.error(f"Error reading log file {file}: {str(e)}")
        
        # Calculate metrics
        num_predictions = len(timestamps)
        
        if not timestamps:
            logger.warning(f"No predictions found in the last {days} days")
            return {"total_predictions": 0}
        
        timestamps.sort()
        
        metrics = {
            "total_predictions": num_predictions,
            "first_prediction": timestamps[0].isoformat(),
            "last_prediction": timestamps[-1].isoformat(),
            "time_period": f"last_{days}_days",
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate predictions per hour if we have enough data
        if len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if time_span > 0:
                metrics["predictions_per_hour"] = round(num_predictions / time_span, 2)
        
        logger.info(f"Volume metrics: {num_predictions} predictions in the last {days} days")
        
        return metrics
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle."""
        logger.info("Starting monitoring cycle")
        
        # Check volume metrics
        volume_metrics = self.check_volume_metrics()
        
        # Calculate performance if feedback data is available
        performance_metrics = self.calculate_performance_metrics()
        
        # Detect data drift if drift detector is available
        drift_results = None
        if self.drift_detector is not None:
            try:
                drift_results = self.drift_detector.detect_drift()
            except Exception as e:
                logger.error(f"Error detecting drift: {str(e)}")
        
        # Combine all monitoring results
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "volume_metrics": volume_metrics,
            "performance_metrics": performance_metrics,
            "drift_results": drift_results
        }
        
        # Save monitoring report
        report_path = os.path.join(
            self.config.LOGS_DIR, 
            f"monitoring_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(monitoring_results, f, indent=4)
        
        logger.info(f"Monitoring report saved to {report_path}")
        
        # Check if retraining is needed
        needs_retraining = False
        
        if drift_results and drift_results.get("drift_detected", False):
            if drift_results.get("drift_percentage", 0) > 0.3:  # If more than 30% of features drifted
                needs_retraining = True
                logger.warning("Significant drift detected. Retraining recommended.")
        
        if performance_metrics and performance_metrics.get("accuracy", 1.0) < 0.7:
            needs_retraining = True
            logger.warning("Low model performance detected. Retraining recommended.")
        
        return needs_retraining
    
    def start_monitoring(self, interval=None):
        """
        Start continuous monitoring in a background thread.
        
        Args:
            interval: Monitoring interval in seconds
                     Defaults to config value
        """
        if interval is None:
            interval = self.config.MONITORING_INTERVAL
        
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring = False
        
        def monitoring_worker():
            logger.info(f"Starting monitoring thread with interval {interval} seconds")
            
            while not self.stop_monitoring:
                try:
                    needs_retraining = self.run_monitoring_cycle()
                    
                    if needs_retraining:
                        # Try to trigger retraining
                        logger.info("Triggering model retraining")
                        try:
                            # You could implement actual retraining trigger here
                            # For now, just log the event
                            with open(os.path.join(self.config.LOGS_DIR, "retraining_triggers.log"), 'a') as f:
                                f.write(f"{datetime.now().isoformat()}: Retraining triggered by monitoring\n")
                        except Exception as e:
                            logger.error(f"Error triggering retraining: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {str(e)}")
                
                # Wait for next cycle
                for _ in range(interval):
                    if self.stop_monitoring:
                        break
                    time.sleep(1)
        
        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring thread started")
    
    def stop_monitoring_thread(self):
        """Stop the monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("No monitoring thread running")
            return
        
        logger.info("Stopping monitoring thread")
        self.stop_monitoring = True
        self.monitoring_thread.join(timeout=10)
        
        if self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not stop cleanly")
        else:
            logger.info("Monitoring thread stopped")
            self.monitoring_thread = None
