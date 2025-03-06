# 10. Deployment Component
class DeploymentComponent(StackComponent):
    """
    Model deployment component.
    """
    
    def initialize(self):
        """Initialize deployment."""
        self.deployment_type = self.config.get('type', 'bentoml')
        
    def deploy_model(self, model, name, version=None):
        """Deploy a model."""
        if self.deployment_type == 'bentoml':
            try:
                import bentoml
                bentoml.save(model, name)
                # More complex deployment would happen here
                return f"Model {name} saved to BentoML repository"
            except ImportError:
                print("Warning: BentoML not installed. Deployment disabled.")
                return None
                
        elif self.deployment_type == 'tf-serving':
            try:
                import tensorflow as tf
                # Save the model in SavedModel format
                export_path = f"./exported_models/{name}/{version or '1'}"
                tf.saved_model.save(model, export_path)
                return f"Model {name} saved for TF Serving at {export_path}"
            except ImportError:
                print("Warning: TensorFlow not installed. Deployment disabled.")
                return None



# Model Deployer Component
class ModelDeployer(StackComponent):
    """Model deployment component."""
    
    def initialize(self):
        """Initialize the model deployer."""
        self.deployer_type = self.config.get('type', 'bentoml')
        
    def deploy_model(self, model, name, version=None):
        """Deploy a model."""
        version = version or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        if self.deployer_type == 'bentoml':
            try:
                import bentoml
                bentoml.save(model, name)
                print(f"Model {name} saved to BentoML repository")
                return f"bentoml://{name}/{version}"
            except ImportError:
                print("Warning: BentoML not installed. Deployment disabled.")
                return None
                
        elif self.deployer_type == 'mlflow':
            try:
                import mlflow
                mlflow.sklearn.log_model(model, name)
                print(f"Model {name} logged to MLflow")
                return f"mlflow://{mlflow.active_run().info.run_id}/{name}"
            except ImportError:
                print("Warning: MLflow not installed. Deployment disabled.")
                return None