import optuna
from ray import tune

class HyperparameterOptimizer:
    def __init__(self, config):
        self.config = config
        self.backend = config.get("tuning.backend", "optuna")
        
    def optimize(self, objective_func, search_space):
        if self.backend == "optuna":
            return self._optuna_optimize(objective_func, search_space)
        elif self.backend == "ray":
            return self._ray_optimize(objective_func, search_space)
    
    def _optuna_optimize(self, objective_func, search_space):
        study = optuna.create_study()
        study.optimize(objective_func, n_trials=self.config.get("tuning.n_trials", 100))
        return study.best_params
    
    def _ray_optimize(self, objective_func, search_space):
        analysis = tune.run(
            objective_func,
            config=search_space,
            num_samples=self.config.get("tuning.n_trials", 100)
        )
        return analysis.get_best_config(metric="accuracy", mode="max")