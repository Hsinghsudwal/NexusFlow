class Stack:
    """
    Container for MLOps stack components.
    """
    
    def __init__(self):
        self.components = {}
        
    def add_component(self, name: str, component: StackComponent):
        """Add a component to the stack."""
        self.components[name] = component
        setattr(self, name, component)
        
    def __getattr__(self, name):
        if name in self.components:
            return self.components[name]
        raise AttributeError(f"Stack has no component '{name}'")


# 6. Stack Registry (ZenML has a similar concept)
class StackRegistry:
    """Registry to keep track of available stacks."""
    
    def __init__(self):
        self.stacks = {}
        self.active_stack = None
        
    def register_stack(self, stack: Stack):
        """Register a stack."""
        self.stacks[stack.name] = stack
        
    def get_stack(self, name: str) -> Optional[Stack]:
        """Get a stack by name."""
        return self.stacks.get(name)
        
    def set_active_stack(self, name: str):
        """Set the active stack."""
        if name not in self.stacks:
            raise ValueError(f"Stack '{name}' not found in registry")
            
        self.active_stack = self.stacks[name]
        
    def get_active_stack(self) -> Optional[Stack]:
        """Get the active stack."""
        return self.active_stack

