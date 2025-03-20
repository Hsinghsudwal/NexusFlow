



from integrations.mlflow.mlflow_tracking import MLflowTracking
from core.pipeline_versioning import PipelineVersioner

def train_model(config):
    # Initialize components
    versioner = PipelineVersioner()
    tracker = MLflowTracking(config)
    
    # Start tracked run
    with tracker.start_run(run_name=versioner.run_id):
        # Log version info
        tracker.log_params(versioner.get_version_info())
        
        # Training logic
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Log model and metrics
        tracker.log_model(model, "model")
        tracker.log_metrics({"accuracy": accuracy_score(y_test, model.predict(X_test))})