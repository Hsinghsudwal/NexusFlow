# experiments/experiment_tracker.py

class ExperimentTracker:
    def __init__(self):
        self.experiment_data = {}

    def log_experiment(self, experiment_id, parameters, metrics, model):
        self.experiment_data[experiment_id] = {
            'parameters': parameters,
            'metrics': metrics,
            'model': model
        }
    
    def get_experiment(self, experiment_id):
        return self.experiment_data.get(experiment_id)
