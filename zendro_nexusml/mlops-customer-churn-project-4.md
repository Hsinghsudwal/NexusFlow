### 8. `src/monitoring/data_drift.py` - Data Drift Detection Implementation (continued)

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
                "drift_detected": js_distance > 0.1  # Threshold for categorical drift
            }
            
            if js_distance > 0.1:
                drift_results["drifted_features"].append(col)
                drift_results["drift_detected"] = True
    
    return drift_results

def calculate_distribution_difference(dist1, dist2):
    """
    Calculate difference between two probability distributions.
    Simple implementation of Jensen-Shannon distance.
    
    Args:
        dist1: First distribution as dictionary
        dist2: Second distribution as dictionary
        
    Returns:
        distance: Distribution difference score
    """
    # Get all unique keys
    all_keys = set(list(dist1.keys()) + list(dist2.keys()))
    
    # Initialize distributions with zeros for missing keys
    dist1_full = {k: dist1.get(k, 0) for k in all_keys}
    dist2_full = {k: dist2.get(k, 0) for k in all_keys}
    
    # Calculate sum of squared differences
    squared_diff_sum = sum((dist1_full[k] - dist2_full[k])**2 for k in all_keys)
    
    # Return distance
    return np.sqrt(squared_diff_sum)

def trigger_drift_check(config):
    """
    Trigger data drift detection check.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        bool: True if drift detected, else False
    """
    # Load reference data
    reference_data_path = config.get("reference_data_path", "data/reference/reference_data.csv")
    if not os.path.exists(reference_data_path):
        logger.warning("Reference data not found. Cannot check for drift.")
        return False
    
    reference_data = pd.read_csv(reference_data_path)
    
    # Load current data
    current_data_path = config.get("current_data_path", "data/current/current_data.csv")
    if not os.path.exists(current_data_path):
        logger.warning("Current data not found. Cannot check for drift.")
        return False
    
    current_data = pd.read_csv(current_data_path)
    
    # Load feature metadata to get column types
    production_dir = "models/production"
    feature_metadata_path = os.path.join(production_dir, "feature_metadata.json")
    
    if os.path.exists(feature_metadata_path):
        feature_metadata = load_metadata(feature_metadata_path)
        numerical_columns = [col for col in feature_metadata.get("features", []) 
                            if col in reference_data.columns and 
                            reference_data[col].dtype in ['int64', 'float64']]
        categorical_columns = [col for col in feature_metadata.get("features", []) 
                              if col in reference_data.columns and 
                              reference_data[col].dtype not in ['int64', 'float64']]
    else:
        numerical_columns = None
        categorical_columns = None
    
    # Detect drift
    drift_results = detect_data_drift(
        reference_data, 
        current_data,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )
    
    # Save drift results
    drift_path = os.path.join("artifacts", "drift", f"drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(drift_path), exist_ok=True)
    save_metadata(drift_results, drift_path)
    
    logger.info(f"Data drift check completed. Results saved to {drift_path}")
    logger.info(f"Drift detected: {drift_results['drift_detected']}")
    if drift_results['drift_detected']:
        logger.info(f"Drifted features: {drift_results['drifted_features']}")
    
    return drift_results["drift_detected"]
```

### 9. `src/monitoring/model_performance.py` - Model Performance Monitoring

```python
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
from src.utils.metadata import load_metadata, save_metadata
import logging

logger = logging.getLogger(__name__)

def setup_monitoring(model_metadata, feature_metadata, monitoring_config):
    """
    Setup monitoring for a deployed model.
    
    Args:
        model_metadata: Model metadata
        feature_metadata: Feature metadata
        monitoring_config: Monitoring configuration
        
    Returns:
        monitoring_setup: Monitoring setup configuration
    """
    # Default monitoring thresholds
    default_thresholds = {
        "performance_drop_threshold": 0.05,  # 5% drop in performance
        "data_drift_threshold": 0.1,         # 10% drift in data
        "prediction_drift_threshold": 0.1,    # 10% drift in predictions
        "monitoring_frequency": "daily",      # daily monitoring
        "alert_emails": []                    # no emails by default
    }
    
    # Merge with provided config
    thresholds = {**default_thresholds, **monitoring_config}
    
    # Setup monitoring configuration
    monitoring_setup = {
        "model_version": model_metadata.get("model_version", "unknown"),
        "model_type": model_metadata.get("model_type", "unknown"),
        "monitoring_start_time": datetime.now().isoformat(),
        "baseline_metrics": model_metadata.get("training_metrics", {}),
        "feature_distribution": feature_metadata.get("feature_stats", {}),
        "thresholds": thresholds,
        "monitoring_log": []
    }
    
    return monitoring_setup

def check_model_performance(prediction_logs, actual_labels, monitoring_config):
    """
    Check model performance based on recent predictions and actuals.
    
    Args:
        prediction_logs: Dataframe with prediction logs
        actual_labels: Dataframe with actual labels
        monitoring_config: Monitoring configuration
        
    Returns:
        performance_check: Performance check results
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    # Merge predictions with actuals
    merged_data = pd.merge(
        prediction_logs, 
        actual_labels, 
        on="prediction_id",  # or another key column
        how="inner"
    )
    
    # If no data or very little data, return null result
    if len(merged_data) < 10:
        logger.warning("Insufficient data for performance monitoring")
        return {
            "status": "insufficient_data",
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(merged_data)
        }
    
    # Calculate performance metrics
    current_metrics = {
        "auc_roc": roc_auc_score(merged_data["actual"], merged_data["probability"]),
        "precision": precision_score(merged_data["actual"], merged_data["prediction"]),
        "recall": recall_score(merged_data["actual"], merged_data["prediction"]),
        "f1_score": f1_score(merged_data["actual"], merged_data["prediction"])
    }
    
    # Load baseline metrics
    baseline_metrics = monitoring_config.get("baseline_metrics", {})
    
    # Calculate performance drops
    performance_drops = {
        metric: baseline_metrics.get(metric, 0) - current_metrics.get(metric, 0)
        for metric in baseline_metrics.keys()
    }
    
    # Check if any metric dropped below threshold
    performance_drop_threshold = monitoring_config.get("thresholds", {}).get("performance_drop_threshold", 0.05)
    performance_alert = any(drop > performance_drop_threshold for drop in performance_drops.values())
    
    performance_check = {
        "status": "alert" if performance_alert else "normal",
        "timestamp": datetime.now().isoformat(),
        "sample_size": len(merged_data),
        "current_metrics": current_metrics,
        "baseline_metrics": baseline_metrics,
        "performance_drops": performance_drops,
        "alert": performance_alert
    }
    
    return performance_check

def log_predictions(model_version, features, predictions, prediction_ids=None):
    """
    Log model predictions for monitoring.
    
    Args:
        model_version: Model version
        features: Input features
        predictions: Model predictions
        prediction_ids: Unique IDs for predictions
        
    Returns:
        log_path: Path to saved prediction logs
    """
    # Create prediction log dataframe
    prediction_log = pd.DataFrame(features)
    
    # Add predictions
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2 and predictions.shape[1] >= 2:
        # For probability predictions from predict_proba
        prediction_log["probability"] = predictions[:, 1]
        prediction_log["prediction"] = (predictions[:, 1] >= 0.5).astype(int)
    else:
        # For class predictions from predict
        prediction_log["prediction"] = predictions
        prediction_log["probability"] = 0.0  # Default
    
    # Add timestamp and IDs
    prediction_log["timestamp"] = datetime.now().isoformat()
    prediction_log["model_version"] = model_version
    
    if prediction_ids is not None:
        prediction_log["prediction_id"] = prediction_ids
    else:
        prediction_log["prediction_id"] = [f"pred_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
                                         for i in range(len(prediction_log))]
    
    # Save prediction logs
    log_dir = os.path.join("logs", "predictions", model_version)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    prediction_log.to_csv(log_path, index=False)
    
    return log_path

def update_monitoring_log(monitoring_results):
    """
    Update monitoring log with latest results.
    
    Args:
        monitoring_results: Latest monitoring results
        
    Returns:
        updated_log_path: Path to updated monitoring log
    """
    production_dir = "models/production"
    monitoring_config_path = os.path.join(production_dir, "monitoring_config.json")
    
    if os.path.exists(monitoring_config_path):
        monitoring_config = load_metadata(monitoring_config_path)
    else:
        logging.error("Monitoring configuration not found")
        return None
    
    # Update log with latest results
    monitoring_config["monitoring_log"].append(monitoring_results)
    
    # Keep only the most recent 100 entries to prevent file growth
    monitoring_config["monitoring_log"] = monitoring_config["monitoring_log"][-100:]
    
    # Update last check time
    monitoring_config["last_check_time"] = datetime.now().isoformat()
    
    # Save updated config
    save_metadata(monitoring_config, monitoring_config_path)
    
    return monitoring_config_path
```

### 10. `src/deployment/model_server.py` - Model Server Implementation

```python
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import logging
from src.utils.metadata import load_metadata
from src.monitoring.model_performance import log_predictions

logger = logging.getLogger(__name__)

def prepare_model_server(model_path, port=8000):
    """
    Prepare model server configuration.
    
    Args:
        model_path: Path to model directory
        port: Port to run the server on
        
    Returns:
        server_config: Server configuration
    """
    server_config = {
        "model_path": model_path,
        "port": port,
        "host": "0.0.0.0",
        "debug": False,
        "endpoint": "/predict"
    }
    
    return server_config

def create_app(model_path):
    """
    Create Flask application for model serving.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        app: Flask application
    """
    app = Flask(__name__)
    
    # Load model and metadata
    model_file = os.path.join(model_path, "model.joblib")
    feature_metadata_file = os.path.join(model_path, "feature_metadata.json")
    training_metadata_file = os.path.join(model_path, "training_metadata.json")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    model = joblib.load(model_file)
    feature_metadata = load_metadata(feature_metadata_file) if os.path.exists(feature_metadata_file) else {}
    training_metadata = load_metadata(training_metadata_file) if os.path.exists(training_metadata_file) else {}
    
    # Get model version
    model_version = os.path.basename(os.path.normpath(model_path))
    
    # Get feature list
    features = feature_metadata.get("features", [])
    required_features = [f for f in features if f != training_metadata.get("target_column")]
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "model_version": model_version})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get input data
            data = request.json
            
            if not data or not isinstance(data, dict):
                return jsonify({"error": "Invalid input: JSON object expected"}), 400
            
            # Check for batch predictions
            if "instances" in data:
                # Batch prediction
                instances = data["instances"]
                if not isinstance(instances, list):
                    return jsonify({"error": "Instances should be a list"}), 400
                
                # Convert to dataframe
                input_df = pd.DataFrame(instances)
            else:
                # Single prediction
                input_df = pd.DataFrame([data])
            
            # Validate input features
            missing_features = [f for f in required_features if f not in input_df.columns]
            if missing_features:
                return jsonify({
                    "error": f"Missing required features: {missing_features}"
                }), 400
            
            # Make prediction
            try:
                probabilities = model.predict_proba(input_df[required_features])
                predictions = (probabilities[:, 1] >= 0.5).astype(int)
                
                # Log predictions for monitoring
                prediction_ids = data.get("prediction_ids", None)
                log_predictions(
                    model_version=model_version,
                    features=input_df[required_features],
                    predictions=probabilities,
                    prediction_ids=prediction_ids
                )
                
                # Prepare response
                if len(predictions) == 1 and "instances" not in data:
                    # Single prediction response
                    response = {
                        "prediction": int(predictions[0]),
                        "probability": float(probabilities[0, 1]),
                        "model_version": model_version
                    }
                else:
                    # Batch prediction response
                    response = {
                        "predictions": predictions.tolist(),
                        "probabilities": probabilities[:, 1].tolist(),
                        "model_version": model_version
                    }
                
                return jsonify(response)
            
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return jsonify({"error": f"Prediction error: {str(e)}"}), 500
        
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            return jsonify({"error": f"Server error: {str(e)}"}), 500
    
    return app

def run_server(config):
    """
    Run the model server.
    
    Args:
        config: Server configuration
        
    Returns:
        app: Flask application instance
    """
    app = create_app(config["model_path"])
    app.run(
        host=config["host"],
        port=config["port"],
        debug=config["debug"]
    )
    return app
```

### 11. `src/db/local_db.py` - Local Database Implementation

```python
import sqlite3
import os
import json
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "db/mlops.db"

def initialize_db():
    """Initialize the local SQLite database with required tables."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    
    # Training runs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,
        model_type TEXT NOT NULL,
        training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metrics TEXT NOT NULL,
        is_retraining BOOLEAN DEFAULT 0,
        is_production BOOLEAN DEFAULT 0
    )
    ''')
    
    # Model artifacts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
    )
    ''')
    
    # Prediction logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        input_data TEXT NOT NULL,
        prediction REAL NOT NULL,
        probability REAL,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        actual_value REAL,
        feedback TEXT
    )
    ''')
    
    # Monitoring logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS monitoring_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,
        check_type TEXT NOT NULL,
        check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        result TEXT NOT NULL,
        alert_triggered BOOLEAN DEFAULT 0
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

def save_training_run(model_version, model_type, metrics, is_retraining=False):
    """
    Save training run information to the database.
    
    Args:
        model_version: Model version identifier
        model_type: Type of model (e.g., 'random_forest')
        metrics: Dictionary of evaluation metrics
        is_retraining: Boolean indicating if this is a retraining run
        
    Returns:
        run_id: ID of the inserted row
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert metrics to JSON string
    metrics_json = json.dumps(metrics)
    
    cursor.execute('''
    INSERT INTO training_runs (model_version, model_type, training_time, metrics, is_retraining)
    VALUES (?, ?, ?, ?, ?)
    ''', (model_version, model_type, datetime.now().isoformat(), metrics_json, is_retraining))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Training run {model_version} saved to database with ID {run_id}")
    
    return run_id

def save_model_artifact(model_version, artifact_path, artifact_type, metadata=None):
    """
    Save model artifact information to the database.
    
    Args:
        model_version: Model version identifier
        artifact_path: Path to the artifact
        artifact_type: Type of artifact (e.g., 'model', 'metrics', 'plots')
        metadata: Dictionary of metadata about the artifact
        
    Returns:
        artifact_id: ID of the inserted row
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert metadata to JSON string if provided
    metadata_json = json.dumps(metadata) if metadata else None
    
    cursor.execute('''
    INSERT INTO model_artifacts (model_version, artifact_path, artifact_type, created_at, metadata)
    VALUES (?, ?, ?, ?, ?)
    ''', (model_version, artifact_path, artifact_type, datetime.now().isoformat(), metadata_json))
    
    artifact_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Model artifact saved to database with ID {artifact_id}")
    
    return artifact_id

def log_prediction(prediction_id, model_version, input_data, prediction, probability=None):
    """
    Log a prediction to the database.
    
    Args:
        prediction_id: Unique identifier for this prediction
        model_version: Model version used for prediction
        input_data: Input features used for prediction
        prediction: Model prediction output
        probability: Prediction probability (for classification)
        
    Returns:
        log_id: ID of the inserted row
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert input data to JSON string
    input_data_json = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
    
    cursor.execute('''
    INSERT INTO prediction_logs (prediction_id, model_version, input_data, prediction, probability, prediction_time)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (prediction_id, model_version, input_data_json, prediction, probability, datetime.now().isoformat()))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return log_id

def update_prediction_actuals(prediction_id, actual_value, feedback=None):
    """
    Update prediction log with actual values and feedback.
    
    Args:
        prediction_id: Prediction identifier
        actual_value: Actual outcome
        feedback: Optional feedback or notes
        
    Returns:
        success: Boolean indicating if update was successful
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        UPDATE prediction_logs
        SET actual_value = ?, feedback = ?
        WHERE prediction_id = ?
        ''', (actual_value, feedback, prediction_id))
        
        if cursor.rowcount == 0:
            logger.warning(f"No prediction found with ID {prediction_id}")
            success = False
        else:
            success = True
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating prediction: {str(e)}")
        success = False
    finally:
        conn.close()
    
    return success

def log_monitoring_check(model_version, check_type, result, alert_triggered=False):
    """
    Log a monitoring check to the database.
    
    Args:
        model_version: Model version being monitored
        check_type: Type of check (e.g., 'data_drift', 'performance')
        result: Dictionary with check results
        alert_triggered: Boolean indicating if an alert was triggered
        
    Returns:
        log_id: ID of the inserted row
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert result to JSON string
    result_json = json.dumps(result)
    
    cursor.execute('''
    INSERT INTO monitoring_logs (model_version, check_type, check_time, result, alert_triggered)
    VALUES (?, ?, ?, ?, ?)
    ''', (model_version, check_type, datetime.now().isoformat(), result_json, alert_triggered))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Monitoring check logged to database with ID {log_id}")
    
    return log_id

def get_model_metrics_history(model_version=None, limit=10):
    """
    Get historical model metrics from the database.
    
    Args:
        model_version: Optional model version to filter by
        limit: Maximum number of records to return
        
    Returns:
        metrics_history: Dataframe with metrics history
    """
    conn = sqlite3.connect(DB_PATH)
    
    query = '''
    SELECT model_version, model_type, training_time, metrics, is_retraining, is_production
    FROM training_runs
    '''
    
    if model_version:
        query += f" WHERE model_version = '{model_version}'"
    
    query += f" ORDER BY training_time DESC LIMIT {limit}"
    
    metrics_history = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse metrics JSON
    metrics_history['metrics'] = metrics_history['metrics'].apply(json.loads)
    
    return metrics_history

def get_prediction_performance(model_version=None, start_date=None, end_date=None):
    """
    Get prediction performance based on actual values.
    
    Args:
        model_version: Optional model version to filter by
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        performance_data: Dataframe with prediction performance
    """
    conn = sqlite3.connect(DB_PATH)
    
    query = '''
    SELECT model_version, prediction, probability, actual_value, prediction_time
    FROM prediction_logs
    WHERE actual_value IS NOT NULL
    '''
    
    if model_version:
        query += f" AND model_version = '{model_version}'"
    
    if start_date:
        query += f" AND prediction_time >= '{start_date}'"
    
    if end_date:
        query += f" AND prediction_time <= '{end_date}'"
    
    performance_data = pd.read_sql_query(query, conn)
    conn.close()
    
    return performance_data
```

### 12. `src/utils/metadata.py` - Metadata Tracking Utilities

```python
import json
import os
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_metadata(metadata, filepath):
    """
    Save metadata to a JSON file.
    
    Args:
        metadata: Dictionary of metadata
        filepath: Path to save the metadata
        
    Returns:
        success: Boolean indicating if save was successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert non-serializable objects to strings
        def json_serial(obj):
            if isinstance(obj, (datetime)):
                return obj.isoformat()
            try:
                return str(obj)
            except:
                return None
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=json_serial)
        
        logger.debug(f"Metadata saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        return False

def load_metadata(filepath):
    """
    Load metadata from a JSON file.
    
    Args:
        filepath: Path to the metadata file
        
    Returns:
        metadata: Dictionary of metadata
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Metadata file not found: {filepath}")
            return {}
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}

def update_metadata(filepath, new_data):
    """
    Update existing metadata with new data.
    
    Args:
        filepath: Path to the metadata file
        new_data: New data to update
        
    Returns:
        success: Boolean indicating if update was successful
    """
    try:
        # Load existing metadata
        metadata = load_metadata(filepath)
        
        # Update with new data
        metadata.update(new_data)
        
        # Save updated metadata
        return save_metadata(metadata, filepath)
    
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}")
        return False

def list_artifacts(artifact_dir):
    """
    List all artifacts in a directory.
    
    Args:
        artifact_dir: Directory to scan for artifacts
        
    Returns:
        artifacts: List of artifact paths
    """
    artifacts = []