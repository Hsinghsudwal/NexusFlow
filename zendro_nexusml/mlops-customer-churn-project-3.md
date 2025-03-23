### 8. `src/monitoring/data_drift.py` - Data Drift Detection Implementation (continued)

```python
                "distance": js_distance,
                "drift_detected": js_distance > 0.1  # Threshold for categorical drift
            }
            
            if js_distance > 0.1:
                drift_results["drifted_features"].append(col)
                drift_results["drift_detected"] = True
    
    logger.info(f"Data drift detection: drift detected = {drift_results['drift_detected']}")
    if drift_results["drift_detected"]:
        logger.info(f"Drifted features: {drift_results['drifted_features']}")
    
    return drift_results

def calculate_distribution_difference(dist1, dist2):
    """
    Calculate difference between two probability distributions.
    Implements a simplified Jensen-Shannon distance.
    """
    # Get all keys
    all_keys = set(list(dist1.keys()) + list(dist2.keys()))
    
    # Fill missing values with 0
    dist1_complete = {k: dist1.get(k, 0) for k in all_keys}
    dist2_complete = {k: dist2.get(k, 0) for k in all_keys}
    
    # Calculate sum of squared differences (simpler than full JS divergence)
    diff_sum = sum((dist1_complete[k] - dist2_complete[k])**2 for k in all_keys)
    return np.sqrt(diff_sum / len(all_keys))

def trigger_drift_check(config):
    """
    Trigger a data drift check between reference and current data.
    Used to determine if retraining is needed.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        drift_detected: Boolean indicating if drift was detected
    """
    # Load reference data
    reference_data_path = config.get("reference_data_path", "data/reference.csv")
    if not os.path.exists(reference_data_path):
        logger.warning(f"Reference data not found at {reference_data_path}. Cannot check for drift.")
        return False
    
    reference_data = pd.read_csv(reference_data_path)
    
    # Load current data
    current_data_path = config.get("current_data_path", "data/current.csv")
    if not os.path.exists(current_data_path):
        logger.warning(f"Current data not found at {current_data_path}. Cannot check for drift.")
        return False
    
    current_data = pd.read_csv(current_data_path)
    
    # Load feature metadata
    production_dir = "models/production"
    feature_metadata_path = os.path.join(production_dir, "feature_metadata.json")
    
    if os.path.exists(feature_metadata_path):
        feature_metadata = load_metadata(feature_metadata_path)
        feature_columns = feature_metadata.get("features", [])
        
        # Filter data to include only model features
        reference_data = reference_data[feature_columns]
        current_data = current_data[feature_columns]
    
    # Detect drift
    drift_results = detect_data_drift(
        reference_data, 
        current_data,
        categorical_columns=config.get("categorical_features", []),
        numerical_columns=config.get("numerical_features", [])
    )
    
    # Save drift results
    drift_results_path = os.path.join("artifacts", "drift", f"drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(drift_results_path), exist_ok=True)
    save_metadata(drift_results, drift_results_path)
    
    return drift_results["drift_detected"]
```

### 9. `src/monitoring/model_performance.py` - Model Performance Monitoring

```python
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
from src.utils.metadata import save_metadata

logger = logging.getLogger(__name__)

def log_prediction(prediction_id, features, prediction, actual=None, metadata=None):
    """
    Log a prediction for monitoring purposes.
    
    Args:
        prediction_id: Unique identifier for the prediction
        features: Feature values used for prediction
        prediction: Model prediction
        actual: Actual value (if available)
        metadata: Additional metadata
    """
    log_dir = "logs/predictions"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log entry
    log_entry = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "actual": actual,
        "metadata": metadata or {}
    }
    
    # Save to log file
    log_file = os.path.join(log_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return log_entry

def calculate_performance_metrics(predictions_file):
    """
    Calculate performance metrics from logged predictions.
    
    Args:
        predictions_file: Path to predictions log file
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    # Read predictions
    predictions = []
    with open(predictions_file, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Filter predictions with actual values
    predictions_with_actuals = [p for p in predictions if p.get("actual") is not None]
    
    if not predictions_with_actuals:
        logger.warning("No predictions with actual values found")
        return {}
    
    # Extract predictions and actuals
    y_pred = [p["prediction"] for p in predictions_with_actuals]
    y_true = [p["actual"] for p in predictions_with_actuals]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {}
    
    try:
        metrics["accuracy"] = accuracy_score(y_true, [round(p) for p in y_pred])
        metrics["precision"] = precision_score(y_true, [round(p) for p in y_pred])
        metrics["recall"] = recall_score(y_true, [round(p) for p in y_pred])
        metrics["f1_score"] = f1_score(y_true, [round(p) for p in y_pred])
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics

def check_performance_degradation(current_metrics, reference_metrics, thresholds=None):
    """
    Check if model performance has degraded.
    
    Args:
        current_metrics: Current performance metrics
        reference_metrics: Reference performance metrics
        thresholds: Degradation thresholds
        
    Returns:
        degradation: Dictionary with degradation results
    """
    if not thresholds:
        thresholds = {
            "accuracy": 0.05,
            "precision": 0.05,
            "recall": 0.05,
            "f1_score": 0.05,
            "auc_roc": 0.05
        }
    
    degradation = {
        "degraded": False,
        "degraded_metrics": [],
        "degradation_values": {}
    }
    
    for metric, threshold in thresholds.items():
        if metric in current_metrics and metric in reference_metrics:
            diff = reference_metrics[metric] - current_metrics[metric]
            rel_diff = diff / reference_metrics[metric]
            
            degradation["degradation_values"][metric] = {
                "absolute": diff,
                "relative": rel_diff,
                "degraded": rel_diff > threshold
            }
            
            if rel_diff > threshold:
                degradation["degraded"] = True
                degradation["degraded_metrics"].append(metric)
    
    return degradation

def setup_monitoring(model_metadata, feature_metadata, monitoring_config):
    """
    Setup monitoring for a deployed model.
    
    Args:
        model_metadata: Model metadata
        feature_metadata: Feature metadata
        monitoring_config: Monitoring configuration
        
    Returns:
        config: Complete monitoring configuration
    """
    # Create monitoring configuration
    config = {
        "enabled": monitoring_config.get("enabled", True),
        "performance_monitoring": {
            "enabled": monitoring_config.get("performance_monitoring", {}).get("enabled", True),
            "metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
            "thresholds": monitoring_config.get("performance_monitoring", {}).get("thresholds", {
                "accuracy": 0.05,
                "precision": 0.05,
                "recall": 0.05,
                "f1_score": 0.05,
                "auc_roc": 0.05
            })
        },
        "data_drift_monitoring": {
            "enabled": monitoring_config.get("data_drift_monitoring", {}).get("enabled", True),
            "check_frequency": monitoring_config.get("data_drift_monitoring", {}).get("check_frequency", "daily"),
            "reference_data_path": monitoring_config.get("data_drift_monitoring", {}).get("reference_data_path", "data/reference.csv"),
            "drift_thresholds": {
                "numerical": 0.05,  # p-value threshold for KS test
                "categorical": 0.1  # threshold for distribution difference
            }
        },
        "alerting": {
            "enabled": monitoring_config.get("alerting", {}).get("enabled", True),
            "channels": monitoring_config.get("alerting", {}).get("channels", ["log"]),
            "thresholds": {
                "degradation": 0.1,  # Overall degradation threshold
                "drift": 0.3  # Overall drift threshold (percentage of features)
            }
        }
    }
    
    # Set up performance baseline if available
    if "training_metrics" in model_metadata:
        config["performance_monitoring"]["baseline"] = model_metadata["training_metrics"]
    
    # Set up feature monitoring configuration
    feature_stats = feature_metadata.get("feature_stats", {})
    
    # Determine categorical and numerical features
    numerical_features = [col for col, stats in feature_stats.items() 
                          if "mean" in stats and "std" in stats]
    
    all_features = feature_metadata.get("features", [])
    categorical_features = [col for col in all_features if col not in numerical_features]
    
    config["data_drift_monitoring"]["numerical_features"] = numerical_features
    config["data_drift_monitoring"]["categorical_features"] = categorical_features
    
    return config
```

### 10. `src/monitoring/alerting.py` - Alerting System

```python
import json
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config_path=None):
        """Initialize the alert manager with configuration."""
        self.config = {
            "enabled": True,
            "channels": ["log"],
            "email": {
                "enabled": False,
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "sender_email": "alerts@example.com",
                "recipients": ["user@example.com"],
                "username": "",
                "password": ""
            },
            "slack": {
                "enabled": False,
                "webhook_url": ""
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config.update(json.load(f))
        
    def send_alert(self, title, message, level="warning", metadata=None):
        """
        Send an alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error, critical)
            metadata: Additional metadata
        
        Returns:
            success: Boolean indicating if alert was sent successfully
        """
        if not self.config.get("enabled", True):
            logger.info(f"Alerting disabled, not sending alert: {title}")
            return False
        
        alert = {
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        success = False
        
        # Send through each enabled channel
        for channel in self.config.get("channels", ["log"]):
            if channel == "log":
                self._log_alert(alert)
                success = True
            elif channel == "email" and self.config.get("email", {}).get("enabled", False):
                success = self._email_alert(alert)
            elif channel == "slack" and self.config.get("slack", {}).get("enabled", False):
                success = self._slack_alert(alert)
            elif channel == "file":
                success = self._file_alert(alert)
        
        return success
    
    def _log_alert(self, alert):
        """Log the alert using Python logging."""
        log_method = getattr(logger, alert["level"], logger.warning)
        log_method(f"ALERT: {alert['title']} - {alert['message']}")
        return True
    
    def _email_alert(self, alert):
        """Send an email alert."""
        try:
            email_config = self.config.get("email", {})
            
            msg = MIMEMultipart()
            msg["Subject"] = f"[{alert['level'].upper()}] {alert['title']}"
            msg["From"] = email_config.get("sender_email")
            msg["To"] = ", ".join(email_config.get("recipients", []))
            
            # Format email body
            body = f"{alert['message']}\n\n"
            if alert["metadata"]:
                body += "Additional Information:\n"
                for key, value in alert["metadata"].items():
                    body += f"- {key}: {value}\n"
            body += f"\nTimestamp: {alert['timestamp']}"
            
            msg.attach(MIMEText(body, "plain"))
            
            # Connect to SMTP server
            server = smtplib.SMTP(email_config.get("smtp_server"), email_config.get("smtp_port"))
            server.starttls()
            
            # Login if credentials provided
            if email_config.get("username") and email_config.get("password"):
                server.login(email_config.get("username"), email_config.get("password"))
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert['title']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _slack_alert(self, alert):
        """Send a Slack alert."""
        try:
            slack_config = self.config.get("slack", {})
            webhook_url = slack_config.get("webhook_url")
            
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            # Format Slack message
            level_emoji = {
                "info": ":information_source:",
                "warning": ":warning:",
                "error": ":x:",
                "critical": ":rotating_light:"
            }
            
            emoji = level_emoji.get(alert["level"], ":warning:")
            
            payload = {
                "text": f"{emoji} *{alert['title']}*",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{emoji} *{alert['title']}*\n{alert['message']}"
                        }
                    }
                ]
            }
            
            # Add metadata as a separate block if present
            if alert["metadata"]:
                metadata_text = "*Additional Information:*\n"
                for key, value in alert["metadata"].items():
                    metadata_text += f"• *{key}:* {value}\n"
                
                payload["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": metadata_text
                    }
                })
            
            # Add timestamp
            payload["blocks"].append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Timestamp:* {alert['timestamp']}"
                    }
                ]
            })
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert['title']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _file_alert(self, alert):
        """Write alert to a file."""
        try:
            alerts_dir = "logs/alerts"
            os.makedirs(alerts_dir, exist_ok=True)
            
            filename = os.path.join(alerts_dir, f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl")
            
            with open(filename, "a") as f:
                f.write(json.dumps(alert) + "\n")
            
            logger.info(f"Alert written to file: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False
```

### 11. `src/deployment/model_server.py` - Model Server Implementation

```python
import os
import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Optional
import uvicorn
from datetime import datetime
import uuid
from src.utils.metadata import load_metadata
from src.monitoring.model_performance import log_prediction

class PredictionRequest(BaseModel):
    features: Dict[str, Union[float, int, str]]
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: float
    prediction_label: str
    probability: float
    prediction_id: str
    timestamp: str
    model_version: str

def create_app(model_path, model_version=None):
    """Create a FastAPI app for model serving."""
    app = FastAPI(title="Churn Prediction API", 
                  description="API for predicting customer churn",
                  version="1.0.0")
    
    # Load model and metadata
    model = joblib.load(os.path.join(model_path, "model.joblib"))
    
    # Load feature metadata
    feature_metadata = load_metadata(os.path.join(model_path, "feature_metadata.json"))
    required_features = feature_metadata.get("features", [])
    
    # Load training metadata
    training_metadata = load_metadata(os.path.join(model_path, "training_metadata.json"))
    
    if model_version is None:
        model_version = training_metadata.get("timestamp", "unknown")
    
    @app.get("/")
    async def root():
        return {
            "message": "Churn Prediction API",
            "model_version": model_version,
            "model_type": training_metadata.get("model_type", "unknown")
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        try:
            # Get features from request
            features = request.features
            request_id = request.request_id or str(uuid.uuid4())
            
            # Validate features
            missing_features = [f for f in required_features if f not in features]
            if missing_features:
                raise HTTPException(status_code=400, 
                                    detail=f"Missing required features: {missing_features}")
            
            # Create feature vector
            feature_vector = pd.DataFrame([features])
            
            # Ensure all required features are present
            for feature in required_features:
                if feature not in feature_vector.columns:
                    feature_vector[feature] = 0  # Default value
            
            # Reorder columns to match training data
            feature_vector = feature_vector[required_features]
            
            # Make prediction
            prediction_proba = model.predict_proba(feature_vector)[0, 1]
            prediction = int(prediction_proba >= 0.5)
            
            # Create response
            response = {
                "prediction": prediction,
                "prediction_label": "Churn" if prediction == 1 else "No Churn",
                "probability": float(prediction_proba),
                "prediction_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "model_version": model_version
            }
            
            # Log prediction for monitoring
            log_prediction(
                prediction_id=request_id,
                features=features,
                prediction=prediction_proba,
                metadata={
                    "model_version": model_version,
                    "api_request": True
                }
            )
            
            return response
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/feedback")
    async def feedback(request: Dict):
        try:
            prediction_id = request.get("prediction_id")
            actual_value = request.get("actual")
            
            if prediction_id is None or actual_value is None:
                raise HTTPException(status_code=400, 
                                    detail="prediction_id and actual value are required")
            
            # Log feedback
            log_prediction(
                prediction_id=prediction_id,
                features={},  # We don't have features here
                prediction=None,  # We don't have prediction here
                actual=actual_value,
                metadata={
                    "feedback": True,
                    "model_version": model_version
                }
            )
            
            return {"status": "success", "message": "Feedback recorded"}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app

def prepare_model_server(production_dir, config):
    """Prepare the model server for deployment."""
    # Create app
    app = create_app(
        model_path=production_dir,
        model_version=config.get("model_version")
    )
    
    return app

def run_server(production_dir, host="0.0.0.0", port=8000):
    """Run the model server."""
    app = create_app(model_path=production_dir)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model server")
    parser.add_argument("--model-path", type=str, default="models/production",
                        help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    
    args = parser.parse_args()
    
    run_server(args.model_path, args.host, args.port)
```

### 12. `src/db/local_db.py` - Local Database Utilities

```python
import sqlite3
import json
import os
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DB_PATH = "data/mlops.db"

def initialize_db():
    """Initialize the database with required tables."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create training runs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,
        model_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        metrics TEXT NOT NULL,
        parameters TEXT,
        is_retraining BOOLEAN DEFAULT 0
    )
    ''')
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        features TEXT,
        prediction REAL NOT NULL,
        actual REAL,
        feedback_timestamp TEXT
    )
    ''')
    
    # Create drift checks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS drift_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        drift_detected BOOLEAN NOT NULL,
        drifted_features TEXT,
        drift_scores TEXT
    )
    ''')
    
    # Create model deployments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_deployments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        environment TEXT NOT NULL,
        status TEXT NOT NULL,
        metadata TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

def save_training_run(model_version, model_type, metrics, parameters=None, is_retraining=False):
    """Save a training run to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO training_runs (model_version, model_type, timestamp, metrics, parameters, is_retraining)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_version,
        model_type,
        datetime.now().isoformat(),
        json.dumps(metrics),
        json.dumps(parameters) if parameters else None,
        1 if is_retraining else 0
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Training run saved for model version {model_version}")

def save_prediction(prediction_id, model_version, features, prediction, actual=None):
    """Save a prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO predictions (prediction_id, model_version, timestamp, features, prediction, actual)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        prediction_id,
        model_version,
        datetime.now().isoformat(),
        json.dumps(features),
        prediction,
        actual
    ))
    
    conn.commit()
    conn.close()
    
    logger.debug(f"Prediction {prediction_id} saved to database")

def save_drift_check(drift_detected, drifted_features=None, drift_scores=None):
    """Save a drift check result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO drift_checks (timestamp, drift_detected, drifted_features, drift_scores)
    VALUES (?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        1 if drift_detected else 0,
        json.dumps(drifted_features) if drifted_features else None,
        json.dumps(drift_scores) if drift_scores else None
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Drift check saved, drift detected: {drift_detected}")

def save_model_deployment(model_version, environment, status, metadata=None):
    """Save a model deployment record to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO model_deployments (model_version, timestamp, environment, status, metadata)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        model_version,
        datetime.now().isoformat(),
        environment,
        status,
        json.dumps(metadata) if metadata else None
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Model deployment saved for version {model_version} to {environment}")

def get_training_history():
    """Get training history from the database."""
    conn = sqlite3.connect(DB_PATH)
    
    query = '''
    SELECT model_version, model_type, timestamp, metrics, is_retraining
    FROM training_runs
    ORDER BY timestamp DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    
    # Parse metrics
    df['metrics'] = df['metrics'].apply(json.loads)
    
    conn.close()
    
    return df

def get_prediction_history(limit=100):
    """Get prediction history from the database."""
    conn = sqlite3.connect(DB_PATH)
    
    query = f'''
    SELECT prediction_id, model_version, timestamp, prediction, actual
    FROM predictions
    ORDER BY timestamp DESC
    LIMIT {limit}
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_model_performance():
    """Get model performance over time."""
    conn = sqlite3.connect(DB_PATH)
    
    query = '''
    SELECT p.model_version, 
           COUNT(*) as count, 
           AVG(p.prediction) as avg_prediction,
           AVG(CASE WHEN p.actual IS NOT NULL THEN ABS(p.prediction - p.actual) ELSE NULL END) as avg_error
    FROM predictions p
    GROUP BY p.model_version
    ORDER BY p.model_version
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def update_prediction_feedback(prediction_id, actual):
    """Update a prediction with feedback (actual value)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE predictions
    SET actual = ?, feedback_timestamp = ?
    WHERE prediction_id = ?
    ''', (
        actual,
        datetime.now().isoformat(),
        prediction_id
    ))
    
    rows_updated = cursor.rowcount
    conn.commit()
    conn.close()
    
    if rows_updated > 0:
        logger.info(f"Feedback recorded for prediction {prediction_id}")
        return True
    else:
        logger.warning(f"Prediction {prediction_id} not found for feedback")
        return False
```

### 13. `docker/Dockerfile` - Docker Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get