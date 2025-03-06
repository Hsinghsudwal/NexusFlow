# 10. Pipeline Runner - Simple execution engine
class PipelineRunner:
    """Runner for executing pipelines."""
    
    def __init__(self, stack: Stack = None):
        self.stack = stack
        
    def run(self, pipeline: Pipeline, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a pipeline."""
        # Create context
        context = Context(stack=self.stack, params=params or {})
        
        # Use orchestrator if available
        if self.stack and 'orchestrator' in self.stack.components:
            orchestrator = self.stack.get_component('orchestrator')
            return orchestrator.run_pipeline(pipeline, context)
        else:
            # Simple local execution
            return pipeline.run(context)
