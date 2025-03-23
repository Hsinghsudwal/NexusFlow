# orchestrator/pipeline_orchestrator.py

class PipelineOrchestrator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def execute(self, config, data):
        for step in self.pipeline:
            try:
                data = step.execute(config, data)
            except Exception as e:
                print(f"Error in {step.__class__.__name__}: {str(e)}")
                break
