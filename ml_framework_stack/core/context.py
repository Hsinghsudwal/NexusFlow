from typing import Any, Dict
from .pipeline import Pipeline
from .catalog import DataCatalog


class Context:
    """
    Execution context for pipelines.
    """
    def __init__(
        self,
        catalog: DataCatalog,
        stack=None,  # MLOps stack
        params: Dict[str, Any] = None
    ):
        self.catalog = catalog
        self.stack = stack
        self.params = params or {}

    def run(self, pipeline: Pipeline, inputs: Dict[str, Any] = None):
        """Execute a pipeline with the current context."""
        inputs = inputs or {}
        required_outputs = set()
        outputs = {}

        # Discover required outputs for the whole pipeline
        for node in pipeline.nodes:
            required_outputs.update(node.inputs.values())

        # Add inputs to outputs
        outputs.update(inputs)

        # Execute each node in order
        for node in pipeline.nodes:
            node_inputs = {}

            # Gather inputs for this node
            for param_name, catalog_name in node.inputs.items():
                if catalog_name in outputs:
                    node_inputs[param_name] = outputs[catalog_name]
                elif catalog_name in self.catalog:
                    node_inputs[param_name] = self.catalog.load(catalog_name)
                else:
                    raise ValueError(f"Input '{catalog_name}' for node '{node.name}' not found")

            # Execute the node
            if self.stack and hasattr(self.stack, 'tracking'):
                with self.stack.tracking.start_run(node.name):
                    node_outputs = node.run(node_inputs)
            else:
                node_outputs = node.run(node_inputs)

            # Store outputs
            outputs.update(node_outputs)

            # Save outputs to catalog if needed
            for catalog_name in node.outputs.values():
                if catalog_name in self.catalog:
                    self.catalog.save(catalog_name, outputs[catalog_name])

        return outputs




# 9. Pipeline Context - Execution environment
class Context:
    """Execution context for pipelines."""
    
    def __init__(self, stack: Stack = None, params: Dict[str, Any] = None):
        self.stack = stack
        self.params = params or {}
        self.artifacts = {}
        
    def get_param(self, name: str, default=None) -> Any:
        """Get a parameter from the context."""
        return self.params.get(name, default)
        
    def set_artifact(self, name: str, artifact: Any) -> None:
        """Set an artifact in the context."""
        self.artifacts[name] = artifact
        
        # Save to artifact store if available
        if self.stack and 'artifact_store' in self.stack.components:
            store = self.stack.get_component('artifact_store')
            store.save_artifact(artifact, name)
            
    def get_artifact(self, name: str) -> Any:
        """Get an artifact from the context."""
        if name in self.artifacts:
            return self.artifacts[name]
            
        # Try to load from artifact store
        if self.stack and 'artifact_store' in self.stack.components:
            store = self.stack.get_component('artifact_store')
            try:
                artifact = store.load_artifact(name)
                self.artifacts[name] = artifact
                return artifact
            except:
                pass
                
        return None
