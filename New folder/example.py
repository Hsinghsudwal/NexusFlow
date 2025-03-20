# example_usage.py
"""Example usage of the custom MLOps framework."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import our custom MLOps framework
from custom_mlops.core import Pipeline, ArtifactStore, step
from custom_mlops.components import (
    DataLoader, DataSplitter, FeatureEngineeringStep,
    ModelTrainer, ModelEvaluator, MLflowTracker, ModelSerializer
)

# Example: Create a custom preprocessing step
@step(name="PreprocessData")
def preprocess_data(data):
    """Custom preprocessing function."""
    # Remove missing values
    data = data.dropna()
    
    # Convert categorical columns to one-hot encoding
    cat_columns = data.select_dtypes(include=['object']).columns
    if not cat_columns.empty:
        data = pd.get_dummies(data, columns=cat_columns)
    
    return data

# Example: Custom feature engineering transformation
def add_polynomial_features(data):
    """Add polynomial features for numeric columns."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns[:2]  # Limit to first 2 numeric cols
    for col in numeric_cols:
        data[f"{col}_squared"] = data[col] ** 2
    return data

# Create feature engineering step
feature_eng = FeatureEngineeringStep()
feature_eng.add_transformation("polynomial_features", add_polynomial_features)

# Create a pipeline
def create_training_pipeline(data_path, model_save_path="models"):
    """Create and return a complete training pipeline."""
    # Initialize the artifact store
    artifact_store = ArtifactStore("./artifacts")
    
    # Create the pipeline
    pipeline = Pipeline("iris_classification_pipeline")
    pipeline.set_artifact_store(artifact_store)
    
    # Add pipeline steps
    pipeline.add_step(DataLoader(source_type="csv", source_path=data_path))
    pipeline.add_step(preprocess_data)
    pipeline.add_step(feature_eng)
    pipeline.add_step(DataSplitter(test_size=0.3, random_state=42))
    
    # Add model training and evaluation
    pipeline.add_step(
        ModelTrainer(
            model=RandomForestClassifier(n_estimators=100, random_state=42),
            target_col="target"
        )
    )
    pipeline.add_step(ModelEvaluator())
    
    # Add tracking and serialization
    pipeline.add_step(MLflowTracker(experiment_name="iris_classification"))
    pipeline.add_step(ModelSerializer(output_dir=model_save_path))
    
    return pipeline

# Create inference pipeline for new data
def create_inference_pipeline(model_path, data_path):
    """Create and return an inference pipeline."""
    # Load the model
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Initialize the artifact store
    artifact_store = ArtifactStore("./inference_artifacts")
    
    # Create the pipeline
    pipeline = Pipeline("iris_inference_pipeline")
    pipeline.set_artifact_store(artifact_store)
    
    # Add pipeline steps
    pipeline.add_step(DataLoader(source_type="csv", source_path=data_path))
    pipeline.add_step(preprocess_data)
    pipeline.add_step(feature_eng)
    
    # Create prediction step
    @step(name="Predictions")
    def predict(data, model=model):
        X = data  # Assuming all columns are features
        predictions = model.predict(X)
        return {"predictions": predictions}
    
    pipeline.add_step(predict)
    
    return pipeline

# Example execution
if __name__ == "__main__":
    # Sample data path (you would replace this with your actual data path)
    data_path = "./data/iris.csv"
    
    # Create and run the training pipeline
    train_pipeline = create_training_pipeline(data_path)
    results = train_pipeline.run()
    
    print("Training pipeline completed!")
    print(f"Model saved to: {results.get('ModelSerializer.model_path')}")
    
    # Now, let's use the model for inference on new data
    model_path = results.get('ModelSerializer.model_path')
    inference_data_path = "./data/new_samples.csv"
    
    inference_pipeline = create_inference_pipeline(model_path, inference_data_path)
    inference_results = inference_pipeline.run()
    
    print("Inference pipeline completed!")
    print(f"Predictions: {inference_results.get('Predictions.predictions')}")