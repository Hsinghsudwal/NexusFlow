class ABTestingFramework:
    def __init__(self, config):
        self.experiments = {}
        self.config = config
        
    def create_experiment(self, name, variants, traffic_split):
        self.experiments[name] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'results': {}
        }
    
    def get_variant(self, experiment_name):
        return np.random.choice(
            self.experiments[experiment_name]['variants'],
            p=self.experiments[experiment_name]['traffic_split']
        )
    
    def log_result(self, experiment_name, variant, metric_name, value):
        self.experiments[experiment_name]['results'].setdefault(variant, {})
        self.experiments[experiment_name]['results'][variant][metric_name] = value

class BayesianBandit(ABTestingFramework):
    def update_traffic(self, experiment_name):
        # Implement Thompson sampling
        pass