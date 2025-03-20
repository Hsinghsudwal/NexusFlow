import numpy as np

class MultiArmBanditStrategy:
    def __init__(self, models, initial_split=[0.5, 0.5]):
        self.models = models
        self.traffic_split = np.array(initial_split)
        self.performance_metrics = np.ones(len(models))
        
    def update_traffic_split(self):
        total = sum(self.performance_metrics)
        self.traffic_split = self.performance_metrics / total
        
    def select_model(self):
        return np.random.choice(self.models, p=self.traffic_split)
    
    def update_performance(self, model_idx, metric):
        self.performance_metrics[model_idx] = metric
        self.update_traffic_split()