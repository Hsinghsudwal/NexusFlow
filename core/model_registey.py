# Model Registry (stores/model_registry.py)
class ModelRegistry:
    def __init__(self, base_path='models'):
        """
        Manage model versioning and storage
        
        Args:
            base_path (str): Base directory for storing models
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.models: Dict[str, Dict] = {}
    
    def register_model(self, 
                       model: Any, 
                       name: str, 
                       metadata: Dict[str, Any] = None):
        """
        Register a new model version
        
        Args:
            model (Any): Model to register
            name (str): Model name
            metadata (Dict[str, Any], optional): Model metadata
        
        Returns:
            str: Model version identifier
        """
        if name not in self.models:
            self.models[name] = {}
        
        # Generate version based on existing models
        version = len(self.models[name]) + 1
        version_key = f"v{version}"
        
        # Save model
        model_path = os.path.join(
            self.base_path, 
            f"{name}_{version_key}.pkl"
        )
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Store model metadata
        self.models[name][version_key] = {
            'path': model_path,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        
        return version_key
    
    def get_model(self, name: str, version: str = None):
        """
        Retrieve a specific model version
        
        Args:
            name (str): Model name
            version (str, optional): Model version
        
        Returns:
            Any: Retrieved model
        """
        if name not in self.models:
            raise ValueError(f"No models found with name: {name}")
        
        if version is None:
            # Get latest version
            version = sorted(
                self.models[name].keys(), 
                key=lambda v: int(v[1:])
            )[-1]
        
        model_info = self.models[name][version]
        
        with open(model_info['path'], 'rb') as f:
            return pickle.load(f)



# Example usage of the ModelRegistry

# Initialize the model registry
# registry = ModelRegistry(base_path='models')
# Register a new model version
# version = registry.register_model({dummy_model}, name="dummy_model", metadata={"author": "user1"})
# print(f"Model registered with version: {version}")
# Retrieve a specific version of the model
# retrieved_model_v1 = registry.get_model(name="dummy_model", version="v1")
# print(f"Retrieved model v1: {retrieved_model_v1}")