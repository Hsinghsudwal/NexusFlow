# example_pipeline.py
from nexusml.core.step import step
from nexusml.core.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define steps using decorators
@step(name="load_data", outputs=["data"])
def load_data(data_path):
    """Load data from a CSV file."""
    data = pd.read_csv(data_path)
    return data

@step(name="preprocess", outputs=["X", "y"])
def preprocess(data):
    """Preprocess the data and split features and target."""
    # Example preprocessing
    data = data.dropna()
    
    # Extract features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    return X, y

@step(name="split_data", outputs=["X_train", "X_test", "y_train", "y_test"])
def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

@step(name="train_model", outputs=["model"])
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

@step(name="evaluate", outputs=["accuracy", "predictions"])
def evaluate(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

# Define the pipeline
def create_pipeline():
    """Create and return the ML pipeline."""
    pipeline = Pipeline(name="iris_classification", description="Classify Iris flowers")
    
    # Add steps to the pipeline
    pipeline.add_step(load_data)
    pipeline.add_step(preprocess, dependencies=["load_data"])
    pipeline.add_step(split_data, dependencies=["preprocess"])
    pipeline.add_step(train_model, dependencies=["split_data"])
    pipeline.add_step(evaluate, dependencies=["train_model", "split_data"])
    
    return pipeline

def run(params=None):
    """Run the pipeline with the given parameters."""
    params = params or {}
    
    # Default parameters
    default_params = {
        "data_path": "iris.csv",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100
    }
    
    # Update with provided parameters
    default_params.update(params)
    
    # Create and run the pipeline
    pipeline = create_pipeline()
    results = pipeline.run(default_params)
    
    # Save pipeline configuration
    pipeline.save()
    
    # Extract an