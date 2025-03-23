class DeploymentManager:
    def __init__(self, model config, registry):
        self.model = model
        self.config = config
        self.registry = registry
        self.deployment_history = []

    def safe_deploy(self, model_path: str, config: Dict):
        """Safely deploy a model and handle rollbacks if necessary."""
        try:
            previous_model = self.registry.get_latest_production_model()
            new_endpoint = self.deployer.deploy(model_path, config)

            # Validate deployment
            if not self._validate_deployment(new_endpoint):
                raise DeploymentError("Deployment validation failed")
            
            self.registry.promote_model(model_path, "production")
            self.deployment_history.append({
                'timestamp': datetime.now(),
                'model': model_path,
                'status': 'success'
            })
        
        except Exception as e:
            print(f"⚠️ Deployment failed: {str(e)}")
            self.rollback(previous_model)

    def rollback(self, previous_model: str):
        """Rollback to the previous model if deployment fails."""
        print(f"🔙 Rolling back to {previous_model}")
        self.deployer.deploy(previous_model, {})  # Passing empty config for rollback
        self.registry.promote_model(previous_model, "production")

    def _validate_deployment(self, endpoint: str) -> bool:
        """Validate deployment by making a test prediction."""
        test_data = {"test": "data"}  # Example test data
        try:
            response = requests.post(endpoint, json=test_data, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False