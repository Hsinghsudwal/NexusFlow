# Add new imports at the top
from datetime import datetime
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple
import boto3  # For AWS integration example
import prometheus_client  # For Prometheus integration example

# ======================
# 1. Model Serialization
# ======================
class ModelSerializer:
    """Handles model serialization with versioning"""
    def __init__(self, artifact_root: str = "models"):
        self.artifact_root = artifact_root
        os.makedirs(artifact_root, exist_ok=True)
        
    def serialize_model(self, model: Any, metadata: dict) -> str:
        """Serialize model with metadata"""
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.artifact_root}/model_v{version}.pkl"
        
        package = {
            'model': model,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'version': version
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(package, f)
            
        return filename

    def load_model(self, model_path: str) -> Tuple[Any, dict]:
        """Load serialized model with metadata"""
        with open(model_path, 'rb') as f:
            package = pickle.load(f)
        return package['model'], package['metadata']

# ==============================
# 2. Cloud Monitoring Integration
# ==============================
class CloudMonitoringService:
    """Abstract class for cloud monitoring integration"""
    def __init__(self, config: dict):
        self.config = config
        
    def log_metrics(self, metrics: dict):
        raise NotImplementedError
        
    def create_alert(self, alert_name: str, condition: Callable):
        raise NotImplementedError

class AWSCloudWatchMonitor(CloudMonitoringService):
    """AWS CloudWatch implementation"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = boto3.client('cloudwatch',
            region_name=config['region'],
            aws_access_key_id=config['access_key'],
            aws_secret_access_key=config['secret_key'])
            
    def log_metrics(self, metrics: dict):
        namespace = self.config.get('namespace', 'ML/Monitoring')
        metric_data = [{
            'MetricName': name,
            'Value': value,
            'Timestamp': datetime.utcnow()
        } for name, value in metrics.items()]
        
        self.client.put_metric_data(
            Namespace=namespace,
            MetricData=metric_data
        )

class PrometheusMonitor(CloudMonitoringService):
    """Prometheus implementation"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.metrics = {}
        
    def log_metrics(self, metrics: dict):
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = prometheus_client.Gauge(
                    f'ml_model_{name}',
                    f'ML Model {name} metric'
                )
            self.metrics[name].set(value)

# ==============================
# 3. Retraining Triggers
# ==============================
class RetrainingTrigger:
    """Handles retraining decision logic"""
    def __init__(self, config: dict):
        self.config = config
        self.history = []
        
    def add_metrics(self, metrics: dict):
        self.history.append(metrics)
        
    def should_retrain(self) -> bool:
        if len(self.history) < 2:
            return False
            
        # Check accuracy degradation
        current_acc = self.history[-1].get('accuracy', 0)
        prev_acc = self.history[-2].get('accuracy', 0)
        if (prev_acc - current_acc) > self.config.get('accuracy_degradation_threshold', 0.05):
            return True
            
        # Check data drift
        if self.history[-1].get('data_drift_score', 0) > self.config.get('data_drift_threshold', 0.2):
            return True
            
        return False

# ==============================
# 4. Data Drift Detection
# ==============================
class DataDriftDetector:
    """Detects data distribution changes"""
    def __init__(self, reference_data: np.ndarray):
        self.reference_data = reference_data
        self.window_size = 1000  # Default window size
        
    def calculate_drift(self, current_data: np.ndarray, method: str = 'psi') -> float:
        if method == 'psi':
            return self._calculate_psi(current_data)
        elif method == 'kl_divergence':
            return self._calculate_kl_divergence(current_data)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")

    def _calculate_psi(self, current_data: np.ndarray) -> float:
        # Implementation of Population Stability Index
        ref_counts, bin_edges = np.histogram(self.reference_data, bins=10)
        curr_counts, _ = np.histogram(current_data, bins=bin_edges)
        
        ref_probs = ref_counts / len(self.reference_data)
        curr_probs = curr_counts / len(current_data)
        
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        return psi

    def _calculate_kl_divergence(self, current_data: np.ndarray) -> float:
        # Implementation of Kullback-Leibler divergence
        ref_counts, bin_edges = np.histogram(self.reference_data, bins=10)
        curr_counts, _ = np.histogram(current_data, bins=bin_edges)
        
        p = ref_counts / len(self.reference_data)
        q = curr_counts / len(current_data)
        
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# ==========================================
# Updated Pipeline with Integration Points
# ==========================================
class TrainingPipeline:
    def __init__(self, path: str):
        # ... (previous init code)
        # Add new config sections
        self.config.config_dict.update({
            'monitoring': {
                'provider': 'aws',  # or 'prometheus'
                'aws_region': 'us-east-1',
                'namespace': 'ML/ChurnModel'
            },
            'retraining': {
                'accuracy_threshold': 0.85,
                'drift_threshold': 0.2
            }
        })
        
    def model_deployment_task(self):
        self.logger.info("Deploying model...")
        model = self.pipeline_stack.get_artifact("model")
        metrics = self.pipeline_stack.get_artifact("metrics")

        # Serialize model
        serializer = ModelSerializer()
        model_path = serializer.serialize_model(
            model=model,
            metadata={
                'training_date': datetime.now().isoformat(),
                'metrics': metrics
            }
        )
        
        # Deployment logic
        if metrics["accuracy"] >= self.config.get("deployment_threshold"):
            deployment_status = {
                "status": "deployed",
                "version": "1.0.0",
                "model_path": model_path
            }
            # Integrate with model registry here
        else:
            deployment_status = {"status": "rejected"}
        
        return {"deployment_info": deployment_status}

    def monitoring_setup_task(self):
        self.logger.info("Initializing monitoring...")
        config = self.config.get('monitoring', {})
        
        if config.get('provider') == 'aws':
            monitor = AWSCloudWatchMonitor({
                'region': config.get('aws_region'),
                'namespace': config.get('namespace')
            })
        elif config.get('provider') == 'prometheus':
            monitor = PrometheusMonitor(config)
        else:
            monitor = None
            
        # Initialize drift detector
        reference_data = self._load_reference_data()
        drift_detector = DataDriftDetector(reference_data)
        
        return {
            "monitoring_system": {
                'service': monitor,
                'drift_detector': drift_detector
            }
        }

    def retraining_trigger_task(self):
        self.logger.info("Checking retraining conditions...")
        metrics = self.pipeline_stack.get_artifact("metrics")
        monitor = self.pipeline_stack.get_artifact("monitoring_system")
        
        # Add to monitoring system
        monitor['service'].log_metrics(metrics)
        
        # Check data drift
        current_data = self._get_current_data()
        drift_score = monitor['drift_detector'].calculate_drift(current_data)
        
        # Update retraining trigger
        trigger = RetrainingTrigger(self.config.get('retraining', {}))
        trigger.add_metrics({**metrics, 'data_drift_score': drift_score})
        
        return {
            "retraining_status": {
                "needed": trigger.should_retrain(),
                "drift_score": drift_score,
                "last_metrics": metrics
            }
        }

    def _load_reference_data(self) -> np.ndarray:
        # Implement actual data loading
        return np.random.rand(1000)  # Placeholder

    def _get_current_data(self) -> np.ndarray:
        # Implement actual data loading
        return np.random.rand(1000)  # Placeholder

# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    pipeline = TrainingPipeline("data/updated_churn_data.csv")
    
    # Run main pipeline
    if pipeline.run():
        # Check retraining status periodically
        retrain_status = pipeline.pipeline_stack.get_artifact("retraining_status")
        if retrain_status['needed']:
            print("Triggering retraining...")
            # Implement retraining workflow here
            
        # Example monitoring integration
        monitor = pipeline.pipeline_stack.get_artifact("monitoring_system")
        monitor['service'].log_metrics({
            'inference_latency': 0.12,
            'prediction_count': 1500
        })



main.py

# Example standalone usage
# 1. Model Serialization
serializer = ModelSerializer()
model = {"trained_model": "RandomForest"}
model_path = serializer.serialize_model(model, {"accuracy": 0.92})

# 2. Data Drift Detection
reference_data = np.random.normal(0, 1, 1000)
current_data = np.random.normal(0.5, 1, 1000)
detector = DataDriftDetector(reference_data)
print(f"PSI Score: {detector.calculate_drift(current_data, method='psi')}")

# 3. Cloud Monitoring
aws_monitor = AWSCloudWatchMonitor({
    'region': 'us-east-1',
    'namespace': 'ML/Test'
})
aws_monitor.log_metrics({'accuracy': 0.91, 'latency': 0.15})

# 4. Retraining Trigger
trigger = RetrainingTrigger({'accuracy_degradation_threshold': 0.05})
trigger.add_metrics({'accuracy': 0.92})
trigger.add_metrics({'accuracy': 0.86})
print(f"Should retrain: {trigger.should_retrain()}")