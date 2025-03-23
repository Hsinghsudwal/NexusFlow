"""
Model monitoring system for detecting data drift and model performance degradation
"""
import os
import json
import time
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import psycopg2
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Summary, Histogram, CollectorRegistry
import requests
from typing import Dict, List, Tuple, Any, Optional
import joblib
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
def load_config():
    config_path = os.getenv("CONFIG_PATH", "/app/config/monitoring.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Prometheus metrics
registry = CollectorRegistry()
MODEL_PERFORMANCE = Gauge('model_performance', 'Model performance metrics', 
                          ['metric', 'model_version'], registry=registry)
DATA_DRIFT = Gauge('data_drift', 'Data drift metrics by feature', 
                   ['feature', 'drift_type'], registry=registry)
PREDICTIONS_DISTRIBUTION = Gauge('predictions_distribution', 'Distribution of predictions', 
                                ['prediction_class'], registry=registry)
FEATURE_DISTRIBUTIONS = Histogram('feature_distributions', 'Histogram of feature values', 
                                 ['feature'], registry=registry)
ALERT_COUNT = Counter('model_monitoring_alerts', 'Count of monitoring alerts', 
                     ['alert_type', 'severity'], registry=registry)
MONITORING_LATENCY = Summary('monitoring_check_latency_seconds', 'Latency of monitoring checks', 
                            ['check_type'], registry=registry)

# Database connection
def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    conn = psycopg2.connect(
        host=config['database']['host'],
        database=config['database']['name'],
        user=config['database']['user'],
        password=config['database']['password'],
        port=config['database']['port']
    )
    return conn

# MLflow setup
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

class DataDriftDetector:
    """Detect drift in feature distributions between training and production data"""
    
    def __init__(self, reference_data_path: str, drift_threshold: float = 0.05):
        self.reference_data = joblib.load(reference_data_path)
        self.drift_threshold = drift_threshold
        self.numeric_features = [col for col in self.reference_data.columns 
                                if self.reference_data[col].dtype in [np.int64, np.float64]]
        self.categorical_features = [col for col in self.reference_data.columns 
                                    if col not in self.numeric_features]
    
    def detect_numeric_drift(self, production_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Detect drift in numeric features using KS test"""
        drift_metrics = {}
        
        for feature in self.numeric_features:
            ref_data = self.reference_data[feature].dropna()
            prod_data = production_data[feature].dropna()
            
            # Skip if not enough data
            if len(prod_data) < 10:
                continue
                
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(ref_data, prod_data)
            
            # Calculate distribution metrics
            drift_metrics[feature] = {
                'ks_stat': float(ks_stat),
                'p_value': float(p_value),
                'mean_diff': float(abs(ref_data.mean() - prod_data.mean())),
                'std_diff': float(abs(ref_data.std() - prod_data.std())),
                'has_drift': p_value < self.drift_threshold
            }
            
            # Log to Prometheus
            DATA_DRIFT.labels(feature=feature, drift_type='ks_stat').set(ks_stat)
            DATA_DRIFT.labels(feature=feature, drift_type='p_value').set(p_value)
            
            # Create histogram
            if feature in config['monitoring']['histogram_features']:
                hist, bin_edges = np.histogram(prod_data, bins='auto')
                for i, count in enumerate(hist):
                    bin_lower = bin_edges[i]
                    bin_upper = bin_edges[i + 1]
                    bin_label = f"{feature}_{bin_lower:.2f}_{bin_upper:.2f}"
                    FEATURE_DISTRIBUTIONS.labels(feature=bin_label).observe(count)
        
        return drift_metrics
    
    def detect_categorical_drift(self, production_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Detect drift in categorical features using chi-square test"""
        drift_metrics = {}
        
        for feature in self.categorical_features:
            ref_counts = self.reference_data[feature].value_counts(normalize=True).to_dict()
            prod_counts = production_data[feature].value_counts(normalize=True).to_dict()
            
            # Combine all categories
            all_categories = set(list(ref_counts.keys()) + list(prod_counts.keys()))
            
            # Calculate chi-square statistic
            chi2_stat = 0
            for category in all_categories:
                ref_prob = ref_counts.get(category, 0)
                prod_prob = prod_counts.get(category, 0)
                chi2_stat += ((prod_prob - ref_prob) ** 2) / (ref_prob + 1e-10)
            
            # Calculate JS divergence
            js_divergence = self._calculate_js_divergence(ref_counts, prod_counts)
            
            drift_metrics[feature] = {
                'chi2_stat': float(chi2_stat),
                'js_divergence': float(js_divergence),
                'has_drift': js_divergence > self.drift_threshold
            }
            
            # Log to Prometheus
            DATA_DRIFT.labels(feature=feature, drift_type='chi2_stat').set(chi2_stat)
            DATA_DRIFT.labels(feature=feature, drift_type='js_divergence').set(js_divergence)
        
        return drift_metrics
    
    def _calculate_js_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Calculate Jensen-Shannon divergence between two probability distributions"""
        # Get all keys
        all_keys = set(list(p.keys()) + list(q.keys()))
        
        # Initialize distributions with zeros for missing categories
        p_complete = {k: p.get(k, 0) for k in all_keys}
        q_complete = {k: q.get(k, 0) for k in all_keys}
        
        # Convert to arrays
        p_array = np.array(list(p_complete.values()))
        q_array = np.array(list(q_complete.values()))
        
        # Calculate the average distribution
        m_array = (p_array + q_array) / 2
        
        # Calculate KL divergence
        kl_p_m = np.sum(p_array * np.log2(p_array / (m_array + 1e-10) + 1e-10))
        kl_q_m = np.sum(q_array * np.log2(q_array / (m_array + 1e-10) + 1e-10))
        
        # Calculate JS divergence
        return (kl_p_m + kl_q_m) / 2
    
    def detect_drift(self, production_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in all features"""
        numeric_drift = self.detect_numeric_drift(production_data)
        categorical_drift = self.detect_categorical_drift(production_data)
        
        # Combine results
        all_drift = {**numeric_drift, **categorical_drift}
        
        # Overall drift assessment
        features_with_drift = [f for f, metrics in all_drift.items() if metrics['has_drift']]
        drift_ratio = len(features_with_drift) / len(all_drift)
        
        drift_result = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_ratio > config['monitoring']['drift_threshold'],
            'drift_ratio': drift_ratio,
            'features_with_drift': features_with_drift,
            'detailed_metrics': all_drift
        }
        
        return drift_result

class ModelPerformanceMonitor:
    """Monitor model performance using labeled data"""
    
    def __init__(self, model_uri: str, performance_threshold: float = 0.7):
        # Load model
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_version = model_uri.split('/')[-1]
        self.performance_threshold = performance_threshold
        
        # Load baseline metrics if available
        try:
            self.baseline_metrics = self._get_baseline_metrics()
        except Exception as e:
            logger.warning(f"Could not load baseline metrics: {str(e)}")
            self.baseline_metrics = None
    
    def _get_baseline_metrics(self) -> Dict[str, float]:
        """Get baseline metrics from MLflow"""
        client = mlflow.tracking.MlflowClient()
        model_name = config['mlflow']['model_name']
        version = self.model_version
        
        run_id = client.get_model_version(model_name, version).run_id
        run = client.get_run(run_id)
        
        metrics = run.data.metrics
        return {
            'accuracy': metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1': metrics.get('f1', 0.0),
            'roc_auc': metrics.get('roc_auc', 0.0)
        }
    
    def evaluate_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance on new labeled data"""
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'f1': float(f1_score(y, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0
            }
            
            # Log metrics to Prometheus
            for metric_name, value in metrics.items():
                MODEL_PERFORMANCE.labels(metric=metric_name, model_version=self.model_version).set(value)
            
            # Compare with baseline if available
            degradation = {}
            if self.baseline_metrics:
                degradation = {
                    f"{metric}_degradation": self.baseline_metrics[metric] - value 
                    for metric, value in metrics.items()
                }
                
                # Check if model has degraded significantly
                primary_metric = config['monitoring']['primary_metric']
                significant_degradation = (
                    degradation.get(f"{primary_metric}_degradation", 0) > 
                    config['monitoring']['degradation_threshold']
                )
            else:
                significant_degradation = metrics[config['monitoring']['primary_metric']] < self.performance_threshold
            
            # Prediction distribution
            pred_counts = pd.Series(y_pred).value_counts(normalize=True)
            for label, count in pred_counts.items():
                PREDICTIONS_DISTRIBUTION.labels(prediction_class=str(label)).set(count)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version,
                'metrics': metrics,
                'baseline_metrics': self.baseline_metrics,
                'degradation': degradation,
                'significant_degradation': significant_degradation,
                'prediction_distribution': pred_counts.to_dict()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            ALERT_COUNT.labels(alert_type='performance_evaluation_error', severity='high').inc()
            raise

class AlertManager:
    """Send alerts when significant issues are detected"""
    
    def __init__(self):
        self.alert_endpoints = config['alerts']['endpoints']
        self.alert_thresholds = config['alerts']
    
    def send_alert(self, alert_type: str, message: str, details: Dict[str, Any], severity: str = 'medium'):
        """Send alert to configured endpoints"""
        try:
            # Increment alert counter
            ALERT_COUNT.labels(alert_type=alert_type, severity=severity).inc()
            
            # Prepare alert payload
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'details': details
            }
            
            # Send to webhook if configured
            if 'webhook' in self.alert_endpoints:
                requests.post(
                    self.alert_endpoints['webhook'],
                    json=alert,
                    headers={'Content-Type': 'application/json'}
                )
            
            # Log alert
            if severity == 'high':
                logger.error(f"ALERT: {message}")
            else:
                logger.warning(f"ALERT: {message}")
            
            # Store alert in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO monitoring_alerts 
                (alert_type, severity, message, details) 
                VALUES (%s, %s, %s, %s)
                """,
                (alert_type, severity, message, json.dumps(details))
            )
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
            return False

class MonitoringService:
    """Main monitoring service that runs periodic checks"""
    
    def __init__(self):
        # Load monitoring configuration
        self.config = config
        
        # Initialize components
        self.drift_detector = DataDriftDetector(
            reference_data_path=self.config['monitoring']['reference_data_path'],
            drift_threshold=self.config['monitoring']['drift_threshold']
        )
        
        self.performance_monitor = ModelPerformanceMonitor(
            model_uri=self.config['mlflow']['model_uri'],
            performance_threshold=self.config['monitoring']['performance_threshold']
        )
        
        self.alert_manager = AlertManager()
        
        # Start Prometheus metrics server
        start_http_server(int(self.config['prometheus']['port']))
        logger.info(f"Started Prometheus metrics server on port {self.config['prometheus']['port']}")
    
    def get_recent_data(self, days: int = 1) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get recent data for monitoring from the database"""
        try:
            conn = get_db_connection()
            
            # Query recent predictions
            query = f"""
            SELECT 
                p.*, c.churn as actual_churn
            FROM 
                predictions p
            LEFT JOIN 
                customers c ON p.customer_id = c.customer_id
            WHERE 
                p.prediction_timestamp >= NOW() - INTERVAL '{days} days'
            """
            
            # Load data
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Extract features and actual values
            feature_cols = [col for col in df.columns if col not in [
                'prediction_id', 'customer_id', 'prediction_timestamp', 
                'churn_prediction', 'churn_probability', 'actual_churn'
            ]]
            
            X = df[feature_cols]
            
            # Check if we have labeled data
            if 'actual_churn' in df.columns and not df['actual_churn'].isna().all():
                y = df['actual_churn']
                return X, y
            else:
                return X, None
            
        except Exception as e:
            logger.error(f"Error fetching recent data: {str(e)}")
            ALERT_COUNT.labels(alert_type='data_fetch_error', severity='medium').inc()
            return pd.DataFrame(), None
    
    def run_drift_check(self):
        """Run data drift check"""
        try:
            start_time = time.time()
            
            # Get recent data
            recent_data, _ = self.get_recent_data(days=self.config['monitoring']['drift_check_days'])
            
            if recent_data.empty:
                logger.warning("No recent data available for drift check")
                return
            
            # Detect drift
            drift_result = self.drift_detector.detect_drift(recent_data)
            
            # Record latency
            MONITORING_LATENCY.labels(check_type='drift').observe(time.time() - start_time)
            
            # Send alert if drift detected
            if drift_result['drift_detected']:
                self.alert_manager.send_alert(
                    alert_type='data_drift',
                    message=f"Data drift detected in {len(drift_result['features_with_drift'])} features",
                    details=drift_result,
                    severity='high' if drift_result['drift_ratio'] > 0.3 else 'medium'
                )
                
                # Log drift result to MLflow
                with mlflow.start_run(run_name="drift_detection"):
                    mlflow.log_metric("drift_ratio", drift_result['drift_ratio'])
                    mlflow.log_param("features_with_drift", drift_result['features_with_drift'])
                    mlflow.log_dict(drift_result, "drift_result.json")
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error in drift check: {str(e)}")
            ALERT_COUNT.labels(alert_type='drift_check_error', severity='high').inc()
    
    def run_performance_check(self):
        """Run model performance check"""
        try:
            start_time = time.time()
            
            # Get recent labeled data
            recent_data, labels = self.get_recent_data(days=self.config['monitoring']['performance_check_days'])
            
            if recent_data.empty or labels is None:
                logger.warning("No labeled data available for performance check")
                return
            
            # Evaluate performance
            performance_result = self.performance_monitor.evaluate_performance(recent_data, labels)
            
            # Record latency
            MONITORING_LATENCY.labels(check_type='performance').observe(time.time() - start_time)
            
            # Send alert if performance degraded
            if performance_result['significant_degradation']:
                self.alert_manager.send_alert(
                    alert_type='performance_degradation',
                    message=f"Model performance degradation detected",
                    details=performance_result,
                    severity='high'
                )
                
                # Log performance result to MLflow
                with mlflow.start_run(run_name="performance_monitoring"):
                    for