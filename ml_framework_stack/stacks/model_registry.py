# 8. Model Registry Component
class ModelRegistry(StackComponent):
    """
    Model versioning and registry component.
    """
    
    def initialize(self):
        """Initialize model registry."""
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            print("Warning: MLflow not installed. Model registry disabled.")
            self.mlflow = None
            
    def register_model(self, model_uri, name):
        """Register a model in the registry."""
        if self.mlflow:
            return self.mlflow.register_model(model_uri, name)
        return None
        
    def get_model(self, name, stage="Production"):
        """Get a model from the registry."""
        if self.mlflow:
            return self.mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
        return None
