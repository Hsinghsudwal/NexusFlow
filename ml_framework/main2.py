
# 13. Example Usage - How to use the framework ZenML style
def example_mlops_pipeline():
    """Example of how to use the framework in ZenML style."""
    # 1. Define steps using the decorator
    @step
    def load_data(data_path: str) -> Any:
        """Load data from a file."""
        import pandas as pd
        return pd.read_csv(data_path)
    
    @step
    def preprocess(data):
        """Preprocess the data."""
        # Drop missing values
        data = data.dropna()
        
        # Convert categorical variables
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category').cat.codes
            
        return data
    
    @step
    def split_data(data, test_size: float = 0.2):
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        # Assume the last column is the target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    @step
    def train_model(X_train, y_train, n_estimators: int = 100):
        """Train a random forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    @step
    def evaluate_model(model, X_test, y_test):
        """Evaluate the model."""
        from sklearn.metrics import accuracy_score
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {"accuracy": accuracy}
    
    # 2. Define a pipeline
    training_pipeline = Pipeline(name="train_classifier")
    
    # 3. Add steps to the pipeline with connections
    # This is more like ZenML's style of connecting steps
    data = load_data().connect(data_path="data/raw/data.csv")
    processed_data = preprocess().connect(data=data)
    X_train, X_test, y_train, y_test = split_data().connect(data=processed_data, test_size=0.2)
    model = train_model().connect(X_train=X_train, y_train=y_train, n_estimators=100)
    metrics = evaluate_model().connect(model=model, X_test=X_test, y_test=y_test)
    
    # Add all steps to the pipeline
    training_pipeline.add_step(data)
    training_pipeline.add_step(processed_data)
    training_pipeline.add_step(split_data)
    training_pipeline.add_step(model)
    training_pipeline.add_step(metrics)
    
    return training_pipeline


# 14. Creating and running with a stack
def run_example_pipeline():
    """Run the example pipeline with a stack."""
    # 1. Create stack components
    experiment_tracker = ExperimentTracker(
        name="mlflow_tracker",
        config={
            "tracking_uri": "sqlite:///mlruns.db",
            "experiment_name": "model_training"
        }
    )
    
    artifact_store = ArtifactStore(
        name="local_store",
        config={
            "type": "local",
            "path": "./artifacts"
        }
    )
    
    orchestrator = Orchestrator(
        name="local_orchestrator",
        config={
            "type": "local"
        }
    )
    
    model_deployer = ModelDeployer(
        name="bentoml_deployer",
        config={
            "type": "bentoml"
        }
    )
    
    # 2. Create and initialize stack
    stack = Stack(name="local_dev_stack")
    stack.add_component("experiment_tracker", experiment_tracker)
    stack.add_component("artifact_store", artifact_store)
    stack.add_component("orchestrator", orchestrator)
    stack.add_component("model_deployer", model_deployer)
    stack.initialize()
    
    # 3. Register the stack
    registry = StackRegistry()
    registry.register_stack(stack)
    registry.set_active_stack("local_dev_stack")
    
    # 4. Create pipeline
    pipeline = example_mlops_pipeline()
    
    # 5. Create runner with stack
    runner = PipelineRunner(stack=stack)
    
    # 6. Run pipeline
    results = runner.run(
        pipeline=pipeline,
        params={
            "n_estimators": 200,
            "test_size": 0.25
        }
    )
    
    return results


# Example of saving and loading stacks
def stack_config_example():
    """Example of saving and loading stack configurations."""
    # Create a stack
    stack = Stack(name="production_stack")
    
    # Add components
    stack.add_component(
        "experiment_tracker",
        ExperimentTracker(
            name="mlflow_remote",
            config={
                "tracking_uri": "https://mlflow.example.com",
                "experiment_name": "production_models"
            }
        )
    )
    
    stack.add_component(
        "artifact_store",
        ArtifactStore(
            name="s3_store",
            config={
                "type": "s3",
                "bucket": "ml-artifacts",
                "path": "production"
            }
        )
    )
    
    # Save stack configuration
    stack.save("./configs/production_stack.yaml")
    
    # Load stack configuration
    loaded_stack = Stack.load("./configs/production_stack.yaml")
    
    return loaded_stack


# Main function
if __name__ == "__main__":
    print("ZenML-like Framework Example")
    results = run_example_pipeline()
    print("Pipeline execution results:", results)
