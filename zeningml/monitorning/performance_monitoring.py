from prometheus_client import start_http_server, Gauge
import numpy as np

class ModelPerformanceMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics = {
            'accuracy': Gauge('model_accuracy', 'Current model accuracy'),
            'latency': Gauge('prediction_latency', 'Prediction latency in ms')
        }
        start_http_server(self.config.get('monitoring_port', 8000))

    def update_metrics(self, predictions, actuals, latency):
        accuracy = np.mean(np.array(predictions) == np.array(actuals))
        self.metrics['accuracy'].set(accuracy)
        self.metrics['latency'].set(latency)
        
    def check_anomalies(self):
        if self.metrics['accuracy']._value.get() < self.config.get('accuracy_threshold', 0.7):
            self.trigger_alert("Accuracy dropped below threshold")

    def trigger_alert(self, message):
        # Integrate with alert manager (PagerDuty, Slack, etc.)
        print(f"ALERT: {message}")