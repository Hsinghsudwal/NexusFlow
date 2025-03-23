import abc
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from prometheus_client import Counter, Gauge, start_http_server
from sklearn.metrics import accuracy_score

# 🔹 Retraining Controller with Enhanced Logging
class RetrainingController:
    def __init__(self, stack, pipeline_factory):
        self.stack = stack
        self.pipeline_factory = pipeline_factory
        self.retraining_policies = {
            'data_drift': DataDriftPolicy(),
            'performance': PerformanceDecayPolicy(),
            'scheduled': TimeBasedPolicy()
        }

    def evaluate_retraining_needs(self):
        """Evaluate retraining policies and trigger retraining if needed."""
        for policy_name, policy in self.retraining_policies.items():
            if policy.check_condition(self.stack):
                self._trigger_retraining(policy_name)

    def _trigger_retraining(self, trigger_reason):
        """Trigger retraining and log the process."""
        print(f"🔄 [INFO] Retraining triggered due to: {trigger_reason}")
        new_pipeline = self.pipeline_factory.build_pipeline(trigger_reason=trigger_reason)
        self.stack.orchestrator.run(new_pipeline)
        self._validate_new_model()
        self._update_model_registry()

    def _validate_new_model(self):
        """Placeholder for model validation logic after retraining."""
        print("✅ [INFO] Validating new model...")

    def _update_model_registry(self):
        """Placeholder for updating the model registry after validation."""
        print("📌 [INFO] Updating model registry with the new model...")

# 🔹 Retrainer with Alerting Mechanism
class Retrainer:
    def __init__(self, stack, pipeline, monitor):
        self.stack = stack
        self.pipeline = pipeline
        self.monitor = monitor

    def check_and_retrain(self):
        """Trigger retraining if data drift exceeds threshold."""
        drift_score = self.monitor.metrics['data_drift']._value.get()
        if drift_score > 0.5:
            print(f"⚠️ [ALERT] Data drift detected! Score: {drift_score:.2f}. Initiating model retraining...")
            self.stack.orchestrator.run(self.pipeline)

# 🔹 Monitoring Base Class
class Monitoring(abc.ABC):
    @abc.abstractmethod
    def track_metrics(self, metrics: dict):
        pass

    @abc.abstractmethod
    def detect_drift(self, data: any):
        pass

# 🔹 Prometheus Monitoring with Logging
class PrometheusMonitoring(Monitoring):
    type = "prometheus"

    def __init__(self):
        self.metrics = {
            'inference_requests': Counter('inference_requests', 'Total inference requests'),
            'data_drift': Gauge('data_drift', 'Data drift score')
        }
        start_http_server(8000)

    def track_metrics(self, metrics: dict):
        print(f"📊 [METRICS] Sending to Prometheus: {metrics}")

    def detect_drift(self, data: any) -> float:
        drift_score = np.random.random()  # Placeholder for drift detection logic
        print(f"🔍 [INFO] Drift score computed: {drift_score:.2f}")
        return drift_score

# 🔹 Model Performance Monitoring with Detailed Logging
class ProductionMonitor:
    def __init__(self, model_endpoint: str):
        self.endpoint = model_endpoint
        self.performance_metrics = {}

    def track_performance(self, X: pd.DataFrame, y: pd.Series):
        """Monitor performance degradation and log results."""
        predictions = self._get_predictions(X)
        current_accuracy = accuracy_score(y, predictions)

        self.performance_metrics[datetime.now()] = {
            'accuracy': current_accuracy,
            'data_distribution': X.describe().to_dict()
        }

        print(f"📊 [INFO] Model accuracy recorded: {current_accuracy:.2f}")

        if current_accuracy < 0.7:
            self.trigger_alert(f"⚠️ [ALERT] Model accuracy dropped below 70%! Current: {current_accuracy:.2f}")

    def _get_predictions(self, data: pd.DataFrame) -> np.ndarray:
        response = requests.post(self.endpoint, json=data.to_dict())
        return np.array(response.json()['predictions'])

    def trigger_alert(self, message: str):
        """Trigger an alert when issues arise."""
        print(f"🚨 {message}")

# 🔹 Data Drift Detector with Logging
class DataDriftDetector:
    def detect_drift(self, data: any) -> float:
        """Returns a simulated drift score with logging."""
        drift_score = np.random.random()
        print(f"🔍 [INFO] Data drift detected with score: {drift_score:.2f}")
        return drift_score

# 🔹 Central Monitoring System with Alerts & Logs
class MonitoringSystem:
    def __init__(self):
        self.prometheus_monitoring = PrometheusMonitoring()
        self.drift_detector = DataDriftDetector()

    def log_prediction(self, features, prediction):
        """Log predictions and monitor for drift issues."""
        self.prometheus_monitoring.metrics['inference_requests'].inc()
        drift_score = self.drift_detector.detect_drift(features)
        self.prometheus_monitoring.metrics['data_drift'].set(drift_score)

        if drift_score > 0.7:
            self.trigger_alert(f"🚨 [ALERT] High data drift detected! Score: {drift_score:.2f}")

    def trigger_alert(self, message: str):
        """Trigger a structured alert message."""
        print(f"🚨 {message}")
