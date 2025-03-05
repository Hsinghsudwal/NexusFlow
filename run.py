from nexusml.training_pipeline import TrainingPipeline

def main():
    path = "data/churn-train.csv"  # Path to your dataset
    pipeline = TrainingPipeline(path)
    pipeline.pipeflow()  # Run the pipeline

if __name__ == "__main__":
    main()




# Demonstration of advanced features
def main():
    # Initialize logger
    logger = NexusLogger('nexusml_demo')
    
    # Load configuration
    config = ConfigManager()
    config.load_config('config.yaml')
    
    # Artifact and model management
    artifact_store = ArtifactStore()
    model_registry = ModelRegistry()
    
    # Example workflow
    logger.info("Starting NexusML Workflow")
    
    try:
        # Simulated steps (replace with actual ML workflow)
        data = [1, 2, 3, 4, 5]  # Example data
        processed_data = [x * 2 for x in data]
        
        # Save artifacts
        data_artifact_path = artifact_store.save(data, 'raw_data')
        processed_artifact_path = artifact_store.save(processed_data, 'processed_data')
        
        # Train a dummy model
        model = processed_data  # Placeholder for actual model
        model_version = model_registry.register_model(
            model, 
            'example_model', 
            metadata={'accuracy': 0.85}
        )
        
        logger.info(f"Model registered: {model_version}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")

if __name__ == '__main__':
    main()