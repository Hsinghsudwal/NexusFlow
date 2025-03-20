class AlertManager:
    def __init__(self, config):
        self.handlers = []
        self.config = config
        
    def register_handler(self, handler):
        self.handlers.append(handler)
        
    def trigger_alert(self, message):
        for handler in self.handlers:
            handler.send_alert(message)

class PerformanceMonitor:
    def check_metrics(self):
        if self.drift_detected:
            self.alert_manager.trigger_alert("Data drift detected!")
        if self.accuracy < self.threshold:
            self.alert_manager.trigger_alert(f"Accuracy dropped to {self.accuracy}")