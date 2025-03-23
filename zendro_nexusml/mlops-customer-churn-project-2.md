"distance": js_distance,
                "drift_detected": js_distance > 0.1  # Threshold can be adjusted
            }
            
            if js_distance > 0.1:
                drift_results["drifted_features"].append(col)
                drift_results["drift_detected"] = True
    
    return drift_results

def calculate_distribution_difference(dist1, dist2):
    """
    Calculate difference between two distributions.
    
    Args:
        dist1: First distribution as a dictionary {category: frequency}
        dist2: Second distribution as a dictionary {category: frequency}
        
    Returns:
        distance: Distance between distributions
    """
    # Combine all categories
    all_categories = set(list(dist1.keys()) + list(dist2.keys()))
    
    # Fill missing categories with 0
    for cat in all_categories:
        if cat not in dist1:
            dist1[cat] = 0
        if cat not in dist2:
            dist2[cat] = 0
    
    # Calculate Jensen-Shannon Distance (simplified)
    distance = 0
    for cat in all_categories:
        p = dist1[cat]
        q = dist2[cat]
        # Add a small epsilon to avoid log(0)
        m = (p + q) / 2 + 1e-10
        p = p + 1e-10
        q = q + 1e-10
        
        # KL Divergence components
        kl_p_m = p * np.log(p / m)
        kl_q_m = q * np.log(q / m)
        
        # Add to distance
        distance += (kl_p_m + kl_q_m) / 2
    
    return distance

def trigger_drift_check(config):
    """
    Trigger data drift check to determine if retraining is needed.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        drift_detected: Boolean indicating if drift was detected
    """
    # Load reference data (data used for training the production model)
    production_dir = "models/production"
    if not os.path.exists(production_dir):
        logger.warning("No production model found. Cannot check for drift.")
        return False
    
    # Load reference data path from metadata
    metadata_path = os.path.join(production_dir, "feature_metadata.json")
    if not os.path.exists(metadata_path):
        logger.warning("No feature metadata found for production model.")
        return False
    
    # Use reference data path from config or default
    reference_data_path = config.get("reference_data_path", "data/processed/reference_data.csv")
    current_data_path = config.get("current_data_path", "data/processed/current_data.csv")
    
    # Load data
    try:
        reference_data = pd.read_csv(reference_data_path)
        current_data = pd.read_csv(current_data_path)
    except Exception as e:
        logger.error(f"Error loading data for drift detection: {e}")
        return False
    
    # Get feature lists
    feature_metadata = load_metadata(metadata_path)
    numerical_columns = [col for col in feature_metadata.get("features", []) 
                         if col in reference_data.columns and 
                         reference_data[col].dtype in ['int64', 'float64']]
    
    categorical_columns = [col for col in feature_metadata.get("features", []) 
                          if col in reference_data.columns and 
                          reference_data[col].dtype == 'object']
    
    # Detect drift
    drift_results = detect_data_drift(
        reference_data, 
        current_data, 
        categorical_columns, 
        numerical_columns
    )
    
    # Save drift results
    drift_path = os.path.join(production_dir, "drift_results.json")
    save_metadata({
        "timestamp": pd.Timestamp.now().isoformat(),
        "results": drift_results
    }, drift_path)
    
    logger.info(f"Data drift detection completed. Drift detected: {drift_results['drift_detected']}")
    
    return drift_results["drift_detected"]
```

### 9. `src/monitoring/model_performance.py` - Model Performance Monitoring

```python
import pandas as pd
import numpy as np
import os
import json
import logging
from src.utils.metadata import load_metadata, save_metadata

logger = logging.getLogger(__name__)

def setup_monitoring(model_metadata, feature_metadata, monitoring_config):
    """
    Setup monitoring for the deployed model.
    
    Args:
        model_metadata: Metadata for the model
        feature_metadata: Metadata for the features
        monitoring_config: Configuration for monitoring
        
    Returns:
        monitoring_setup: Dictionary with monitoring setup
    """
    # Default monitoring thresholds
    default_thresholds = {
        "performance_drop_pct": 0.05,  # 5% performance drop
        "data_drift_threshold": 0.1,   # Drift threshold
        "prediction_drift_threshold": 0.1,  # Prediction distribution drift
        "monitoring_frequency": "daily",  # Monitoring frequency
        "retraining_trigger": "auto"  # auto or manual
    }
    
    # Update with provided config
    thresholds = {**default_thresholds, **monitoring_config}
    
    # Setup monitoring based on model type and features
    monitoring_setup = {
        "enabled": True,
        "thresholds": thresholds,
        "baseline": {
            "performance": model_metadata.get("training_metrics", {}),
            "feature_distributions": {
                col: {"mean": feature_metadata["feature_stats"][col]["mean"], 
                      "std": feature_metadata["feature_stats"][col]["std"]}
                for col in feature_metadata["feature_stats"]
            }
        },
        "metrics_to_monitor": [
            "accuracy", 
            "auc_roc", 
            "f1_score", 
            "feature_drift"
        ],
        "alerts": {
            "email": monitoring_config.get("alert_email"),
            "slack": monitoring_config.get("slack_webhook"),
            "performance_drop": True,
            "data_drift": True
        }
    }
    
    logger.info(f"Model monitoring setup complete with {len(monitoring_setup['metrics_to_monitor'])} metrics")
    
    return monitoring_setup

def record_prediction_metrics(predictions, actual=None, metadata=None):
    """
    Record metrics for model predictions to track performance over time.
    
    Args:
        predictions: Model predictions
        actual: Actual values (if available)
        metadata: Additional metadata about the prediction batch
        
    Returns:
        recorded: Boolean indicating if metrics were recorded
    """
    if metadata is None:
        metadata = {}
    
    # Prepare monitoring record
    monitoring_record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "prediction_count": len(predictions),
        "prediction_stats": {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "distribution": {
                "0-0.2": sum(1 for p in predictions if p < 0.2) / len(predictions),
                "0.2-0.4": sum(1 for p in predictions if 0.2 <= p < 0.4) / len(predictions),
                "0.4-0.6": sum(1 for p in predictions if 0.4 <= p < 0.6) / len(predictions),
                "0.6-0.8": sum(1 for p in predictions if 0.6 <= p < 0.8) / len(predictions),
                "0.8-1.0": sum(1 for p in predictions if p >= 0.8) / len(predictions)
            }
        },
        "metadata": metadata
    }
    
    # Add performance metrics if actual values are provided
    if actual is not None:
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        # Convert predictions to binary if needed
        binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
        
        monitoring_record["performance"] = {
            "accuracy": accuracy_score(actual, binary_preds),
            "f1_score": f1_score(actual, binary_preds),
            "auc_roc": roc_auc_score(actual, predictions)
        }
    
    # Save monitoring record
    monitoring_dir = "monitoring/prediction_logs"
    os.makedirs(monitoring_dir, exist_ok=True)
    
    filename = f"prediction_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(monitoring_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(monitoring_record, f, indent=2)
    
    logger.info(f"Prediction metrics recorded to {filepath}")
    
    # Check if performance alert should be triggered
    if actual is not None:
        check_performance_alert(monitoring_record)
    
    return True

def check_performance_alert(monitoring_record):
    """
    Check if performance alert should be triggered based on latest metrics.
    
    Args:
        monitoring_record: Latest monitoring record with performance metrics
        
    Returns:
        alert_triggered: Boolean indicating if alert was triggered
    """
    # Load production model monitoring setup
    production_dir = "models/production"
    monitoring_path = os.path.join(production_dir, "monitoring_config.json")
    
    if not os.path.exists(monitoring_path):
        logger.warning("No monitoring configuration found.")
        return False
    
    monitoring_config = load_metadata(monitoring_path)
    
    # Get baseline performance and thresholds
    baseline = monitoring_config["baseline"]["performance"]
    thresholds = monitoring_config["thresholds"]
    
    # Compare current performance with baseline
    current = monitoring_record["performance"]
    alerts = []
    
    # Check each metric
    for metric in ["accuracy", "auc_roc", "f1_score"]:
        if metric in baseline and metric in current:
            # Calculate relative drop
            relative_drop = (baseline[metric] - current[metric]) / baseline[metric]
            
            if relative_drop > thresholds["performance_drop_pct"]:
                alerts.append({
                    "metric": metric,
                    "baseline": baseline[metric],
                    "current": current[metric],
                    "drop_pct": relative_drop * 100
                })
    
    # If alerts, log and trigger notifications
    if alerts:
        alert_record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "alert_type": "performance_drop",
            "alerts": alerts
        }
        
        # Save alert
        alert_dir = "monitoring/alerts"
        os.makedirs(alert_dir, exist_ok=True)
        
        filename = f"performance_alert_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(alert_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(alert_record, f, indent=2)
        
        logger.warning(f"Performance alert triggered: {alerts}")
        
        # Send notifications if configured
        if monitoring_config["alerts"]["email"]:
            # Code to send email alert would go here
            pass
        
        if monitoring_config["alerts"]["slack"]:
            # Code to send Slack alert would go here
            pass
        
        return True
    
    return False
```

### 10. `src/deployment/model_server.py` - Model Serving Implementation

```python
import os
import pandas as pd
import joblib
import json
from flask import Flask, request, jsonify
import logging
from src.monitoring.model_performance import record_prediction_metrics

logger = logging.getLogger(__name__)

def prepare_model_server(model_path, config):
    """
    Prepare the Flask app for serving model predictions.
    
    Args:
        model_path: Path to the model file
        config: Configuration for the model server
        
    Returns:
        app: Flask application
    """
    app = Flask(__name__)
    
    # Load the model
    try:
        model = joblib.load(os.path.join(model_path, "model.joblib"))
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
    
    # Load feature metadata
    try:
        feature_metadata_path = os.path.join(model_path, "feature_metadata.json")
        with open(feature_metadata_path, "r") as f:
            feature_metadata = json.load(f)
        expected_features = feature_metadata.get("features", [])
        logger.info(f"Loaded feature metadata with {len(expected_features)} features")
    except Exception as e:
        logger.error(f"Error loading feature metadata: {e}")
        expected_features = []
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
        return jsonify({"status": "ok", "message": "Model is ready"}), 200
    
    @app.route("/predict", methods=["POST"])
    def predict():
        """Prediction endpoint."""
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
        # Get data from request
        try:
            data = request.get_json()
            
            # Handle single instance or batch
            is_batch = isinstance(data, list)
            if not is_batch:
                data = [data]
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Check for required features
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                return jsonify({
                    "status": "error", 
                    "message": f"Missing features: {missing_features}"
                }), 400
            
            # Make predictions
            predictions = model.predict_proba(df[expected_features])[:, 1]
            
            # Record prediction metrics for monitoring
            record_prediction_metrics(
                predictions=predictions, 
                metadata={"request_id": request.headers.get("X-Request-ID", "unknown")}
            )
            
            # Return predictions
            if is_batch:
                result = {"predictions": predictions.tolist()}
            else:
                result = {"prediction": predictions[0]}
            
            return jsonify(result), 200
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    return app

def run_model_server(model_path, host="0.0.0.0", port=8000):
    """
    Run the model server.
    
    Args:
        model_path: Path to the model file
        host: Host to run the server on
        port: Port to run the server on
    """
    config = {
        "host": host,
        "port": port
    }
    
    app = prepare_model_server(model_path, config)
    app.run(host=host, port=port)
    
    return app
```

### 11. `src/db/local_db.py` - Local Database Implementation

```python
import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Database path
DB_PATH = "db/mlops.db"

def initialize_db():
    """
    Initialize the local SQLite database with the required tables.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT,
        model_type TEXT,
        timestamp DATETIME,
        metrics TEXT,
        is_retraining BOOLEAN,
        tags TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_deployments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT,
        deployment_timestamp DATETIME,
        environment TEXT,
        status TEXT,
        metrics TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS monitoring_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type TEXT,
        model_version TEXT,
        timestamp DATETIME,
        details TEXT,
        severity TEXT,
        resolved BOOLEAN
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT,
        timestamp DATETIME,
        request_id TEXT,
        features TEXT,
        prediction REAL,
        actual_value REAL NULL,
        metadata TEXT
    )
    ''')
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

def save_training_run(model_version, model_type, metrics, is_retraining=False, tags=None):
    """
    Save a training run to the database.
    
    Args:
        model_version: Version of the model
        model_type: Type of model
        metrics: Dictionary of training metrics
        is_retraining: Boolean indicating if this was a retraining run
        tags: Optional tags for the run
    
    Returns:
        run_id: ID of the inserted run
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Prepare data
    if tags is None:
        tags = {}
    
    # Insert training run
    cursor.execute(
        "INSERT INTO training_runs (model_version, model_type, timestamp, metrics, is_retraining, tags) VALUES (?, ?, ?, ?, ?, ?)",
        (
            model_version,
            model_type,
            datetime.now().isoformat(),
            json.dumps(metrics),
            1 if is_retraining else 0,
            json.dumps(tags)
        )
    )
    
    # Get the ID of the inserted run
    run_id = cursor.lastrowid
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.info(f"Training run saved with ID {run_id}")
    
    return run_id

def save_model_deployment(model_version, environment, status, metrics=None):
    """
    Save a model deployment to the database.
    
    Args:
        model_version: Version of the model
        environment: Deployment environment
        status: Deployment status
        metrics: Optional deployment metrics
    
    Returns:
        deployment_id: ID of the inserted deployment
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Prepare data
    if metrics is None:
        metrics = {}
    
    # Insert deployment
    cursor.execute(
        "INSERT INTO model_deployments (model_version, deployment_timestamp, environment, status, metrics) VALUES (?, ?, ?, ?, ?)",
        (
            model_version,
            datetime.now().isoformat(),
            environment,
            status,
            json.dumps(metrics)
        )
    )
    
    # Get the ID of the inserted deployment
    deployment_id = cursor.lastrowid
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.info(f"Model deployment saved with ID {deployment_id}")
    
    return deployment_id

def save_monitoring_alert(alert_type, model_version, details, severity="medium"):
    """
    Save a monitoring alert to the database.
    
    Args:
        alert_type: Type of alert
        model_version: Version of the model
        details: Alert details
        severity: Alert severity
    
    Returns:
        alert_id: ID of the inserted alert
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Insert alert
    cursor.execute(
        "INSERT INTO monitoring_alerts (alert_type, model_version, timestamp, details, severity, resolved) VALUES (?, ?, ?, ?, ?, ?)",
        (
            alert_type,
            model_version,
            datetime.now().isoformat(),
            json.dumps(details),
            severity,
            0  # Not resolved
        )
    )
    
    # Get the ID of the inserted alert
    alert_id = cursor.lastrowid
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.warning(f"Monitoring alert saved with ID {alert_id}")
    
    return alert_id

def log_prediction(model_version, features, prediction, actual_value=None, request_id=None, metadata=None):
    """
    Log a prediction to the database.
    
    Args:
        model_version: Version of the model
        features: Input features
        prediction: Prediction value
        actual_value: Actual value (if available)
        request_id: Request ID
        metadata: Additional metadata
    
    Returns:
        log_id: ID of the inserted log
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Prepare data
    if metadata is None:
        metadata = {}
    
    # Insert prediction log
    cursor.execute(
        "INSERT INTO prediction_logs (model_version, timestamp, request_id, features, prediction, actual_value, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            model_version,
            datetime.now().isoformat(),
            request_id,
            json.dumps(features),
            prediction,
            actual_value,
            json.dumps(metadata)
        )
    )
    
    # Get the ID of the inserted log
    log_id = cursor.lastrowid
    
    # Commit and close
    conn.commit()
    conn.close()
    
    return log_id

def get_model_performance_history(model_version=None, limit=10):
    """
    Get the performance history of a model from the database.
    
    Args:
        model_version: Version of the model (optional)
        limit: Maximum number of records to return
    
    Returns:
        history: List of performance records
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    # Build query
    query = "SELECT * FROM training_runs"
    params = []
    
    if model_version:
        query += " WHERE model_version = ?"
        params.append(model_version)
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    # Execute query
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Process results
    history = []
    for row in rows:
        record = dict(row)
        record["metrics"] = json.loads(record["metrics"])
        record["tags"] = json.loads(record["tags"])
        history.append(record)
    
    # Close connection
    conn.close()
    
    return history
```

### 12. `docker/Dockerfile` - Docker Container Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model \
    API_PORT=8000

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY main.py /app/

# Create directories
RUN mkdir -p /app/model /app/logs /app/monitoring

# Copy model files (these will be mounted at runtime)
COPY ${MODEL_PATH:-models/production}/* /app/model/

# Expose port
EXPOSE ${API_PORT}

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.deployment.model_server"]
CMD ["--model-path", "/app/model", "--port", "${API_PORT}"]
```

### 13. `docker/docker-compose.yml` - Docker Compose Configuration

```yaml
version: '3.8'

services:
  model-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../models/production:/app/model
      - ../logs:/app/logs
      - ../monitoring:/app/monitoring
    environment:
      - MODEL_PATH=/app/model
      - API_PORT=8000
      - LOG_LEVEL=INFO
    restart: unless-stopped
    networks:
      - mlops-network

  localstack:
    image: localstack/localstack:latest
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3,sqs,lambda
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - ../localstack:/docker-entrypoint-initaws.d
      - localstack-data:/tmp/localstack
    networks:
      - mlops-network

  prefect-server:
    image: prefecthq/prefect:2.8.7
    ports:
      - "4200:4200"
    volumes:
      - prefect-data:/root/.prefect
    environment:
      - PREFECT_UI_API_URL=http://localhost:4200/api
    command: ["prefect", "server", "start"]
    networks:
      - mlops-network

  monitoring-dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8050:8050"
    volumes:
      - ../monitoring:/app/monitoring
    environment:
      - DATABASE_URL=sqlite:///app/db/mlops.db
    depends_on:
      - model-api
    networks:
      - mlops-network

networks:
  mlops-network:
    driver: bridge

volumes:
  localstack-data:
  prefect-data:
```

### 14. `.github/workflows/ci.yml` - CI Pipeline

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

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
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pytest-cov flake8
        
    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src/
        
    - name: Build Docker image
      run: |
        docker build -t churn-prediction:test -f docker/Dockerfile .
        
    - name: Test Docker image
      run: |
        docker run --name test-container -d churn-prediction:test
        docker stop test-container
        docker rm test-container
```

### 15. `.github/workflows/cd.yml` - CD Pipeline

```yaml
name: CD Pipeline

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  deploy:
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
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/ --junitxml=test-results.xml
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/Dockerfile
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          ghcr.io/${{ github.repository_owner }}/churn-prediction:latest
          ghcr.io/${{ github.repository_owner }}/churn-prediction:${{ github.sha }}
          
    - name: Deploy to staging
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploying to staging environment"
        # Deployment script or command would go here
        
    - name: Deploy to production
      if: startsWith(github.ref, 'refs/tags/v')
      run: |
        echo "Deploying to production environment"
        # Production deployment script or command would go here
```

## Usage Guide

### Running the Project

1. **Setup environment:**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/mlops-churn.git
   cd mlops-churn
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run the entire pipeline:**
   ```bash
   python main.py --pipeline full
   ```

3. **Run individual pipeline stages:**
   ```bash
   # Data processing only
   python main.py --pipeline data
   
   # Model training only
   python main.py --pipeline train
   
   # Model evaluation only
   python main.py --pipeline evaluate
   
   # Model deployment only
   python main.py --pipeline deploy
   
   # Re-training pipeline (checks for data drift first)
   python main.py --pipeline retrain
   ```

4. **Run with Docker:**
   ```bash
   # Build and start the containers
   cd docker
   docker-compose up -d
   
   # Check the running services
   docker-compose ps
   
   # Stop the containers
   docker-compose down
   ```

### Retraining Pipeline

The retraining pipeline automatically detects data drift and retrains the model if necessary:

1. **Trigger retraining:**
   ```bash
   python main.py --pipeline retrain
   ```

2. **Scheduled retraining using cron:**
   ```bash
   # Example cron job to run retraining weekly
   0 0 * * 0 cd /path/to/mlops-churn && python main.py --pipeline retrain
   ```

### Monitoring

The monitoring system tracks:
- Model performance metrics
- Data drift detection
- Prediction logs and patterns
- Alerts for performance degradation

You can access the monitoring dashboard at `http://localhost:8050` when running with Docker.

## Key Features

1. **Modular Pipeline Architecture**
   - Independent, composable pipeline components
   - Clean separation of concerns
   - Easy to extend and modify

2. **Comprehensive Monitoring**
   - Data drift detection
   - Model performance tracking