# 7. Experiment Tracking Component
class ExperimentTracker(StackComponent):
    """
    MLflow-based experiment tracking component.
    """
    
    def initialize(self):
        """Initialize MLflow tracking."""
        try:
            import mlflow
            self.mlflow = mlflow
            
            # Configure MLflow
            tracking_uri = self.config.get('tracking_uri')
            experiment_name = self.config.get('experiment_name', 'default')
            
            if tracking_uri:
                self.mlflow.set_tracking_uri(tracking_uri)
                
            self.mlflow.set_experiment(experiment_name)
            
        except ImportError:
            print("Warning: MLflow not installed. Experiment tracking disabled.")
            self.mlflow = None
            
    def start_run(self, name=None):
        """Start a new run."""
        if self.mlflow:
            return self.mlflow.start_run(run_name=name)
        return DummyContext()
        
    def log_params(self, params):
        """Log parameters."""
        if self.mlflow:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
                
    def log_metrics(self, metrics):
        """Log metrics."""
        if self.mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value)
                
    def log_model(self, model, name):
        """Log a model."""
        if self.mlflow:
            self.mlflow.sklearn.log_model(model, name)


# 7. MLOps Stack Component Implementations (ZenML style)

# Experiment Tracker Component
class ExperimentTracker(StackComponent):
    """MLflow-based experiment tracking component."""
    
    def initialize(self):
        """Initialize MLflow tracking."""
        try:
            import mlflow
            self.mlflow = mlflow
            
            # Configure MLflow
            tracking_uri = self.config.get('tracking_uri')
            experiment_name = self.config.get('experiment_name', 'default')
            
            if tracking_uri:
                self.mlflow.set_tracking_uri(tracking_uri)
                
            self.mlflow.set_experiment(experiment_name)
            
        except ImportError:
            print("Warning: MLflow not installed. Experiment tracking disabled.")
            self.mlflow = None
            
    def start_run(self, run_name=None):
        """Start a new run."""
        if self.mlflow:
            return self.mlflow.start_run(run_name=run_name)
        return DummyContext()
        
    def log_params(self, params):
        """Log parameters."""
        if self.mlflow:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
                
    def log_metrics(self, metrics):
        """Log metrics."""
        if self.mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value)
                
    def log_model(self, model, name):
        """Log a model."""
        if self.mlflow:
            self.mlflow.sklearn.log_model(model, name)
