from typing import Any


class DataCatalog:
    """
    A data catalog manages the loading and saving of datasets.
    """
    def __init__(self):
        self._datasets = {}

    def add(self, name: str, dataset):
        """Add a dataset to the catalog."""
        self._datasets[name] = dataset

    def load(self, name: str) -> Any:
        """Load a dataset from the catalog."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found in the catalog")

        return self._datasets[name].load()

    def save(self, name: str, data: Any):
        """Save data to a dataset in the catalog."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found in the catalog")

        self._datasets[name].save(data)

    def __contains__(self, name: str) -> bool:
        return name in self._datasets




# 5. Stack - ZenML's approach to MLOps integrations
class Stack:
    """
    Container for MLOps stack components.
    Similar to ZenML's stack concept.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.uuid = str(uuid.uuid4())
        self.components = {}
        
    def add_component(self, component_type: str, component: StackComponent):
        """Add a component to the stack."""
        self.components[component_type] = component
        return self
        
    def get_component(self, component_type: str) -> Optional[StackComponent]:
        """Get a component from the stack by type."""
        return self.components.get(component_type)
        
    def initialize(self):
        """Initialize all components in the stack."""
        for component in self.components.values():
            component.initialize()
            
    @classmethod
    def load(cls, stack_path: str) -> "Stack":
        """Load a stack configuration from a YAML file."""
        with open(stack_path, 'r') as f:
            config = yaml.safe_load(f)
            
        stack = cls(name=config.get('name', 'default_stack'))
        
        # Add components based on configuration
        # This would normally involve a registry of component types
        # but simplified here
        
        return stack
        
    def save(self, stack_path: str):
        """Save stack configuration to a YAML file."""
        config = {
            'name': self.name,
            'uuid': self.uuid,
            'components': {
                component_type: {
                    'name': component.name,
                    'type': component.__class__.__name__,
                    'config': component.config
                }
                for component_type, component in self.components.items()
            }
        }
        
        with open(stack_path, 'w') as f:
            yaml.dump(config, f)
