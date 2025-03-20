class RollbackManager:
    def __init__(self, registry):
        self.registry = registry
        self.deployment_history = []
        
    def deploy(self, model, version):
        self.deployment_history.append({
            "timestamp": datetime.now(),
            "version": version,
            "model": model
        })
        self.registry.promote_model(model.name, version)
        
    def rollback(self, steps=1):
        previous_version = self.deployment_history[-1-steps]
        self.deploy(previous_version["model"], previous_version["version"])
        return previous_version

class AutoRollback:
    def __init__(self, monitor, registry, threshold=0.1):
        self.monitor = monitor
        self.registry = registry
        self.threshold = threshold
        
    def check_and_rollback(self):
        current_perf = self.monitor.get_current_performance()
        previous_perf = self.monitor.get_historical_performance()
        
        if (previous_perf - current_perf) > self.threshold:
            self.registry.rollback()
            return True
        return False