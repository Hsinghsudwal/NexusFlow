"""Example pipeline using NexusML."""
from nexusml import Pipeline, Step
from nexusml.core.metrics import ModelEvaluator
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from nexusml.storage.cloud import S3StorageProvider
import os

logger = logging.getLogger(__name__)

def data_loading(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Load sample data."""
    logger.info("Loading and splitting data...")
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def preprocessing(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Preprocess data."""
    logger.info("Preprocessing features...")
    X_train = artifacts["X_train"]
    X_test = artifacts["X_test"]
    y_train = artifacts["y_train"]
    y_test = artifacts["y_test"]

    # Normalize features
    X_train_normalized = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test_normalized = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)  # Use training stats

    return {
        "X_train_processed": X_train_normalized,
        "X_test_processed": X_test_normalized,
        "y_train": y_train,
        "y_test": y_test
    }

def training(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Train a simple model."""
    logger.info("Training model...")
    from sklearn.linear_model import LogisticRegression

    X = artifacts["X_train_processed"]
    y = artifacts["y_train"]

    model = LogisticRegression()
    model.fit(X, y)
    return {"model": model}

def evaluation(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    model = artifacts["model"]
    X_test = artifacts["X_test_processed"]
    y_test = artifacts["y_test"]

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = ModelEvaluator.evaluate_classifier(y_test, y_pred)
    logger.info(f"Model metrics: {metrics}")
    return {"metrics": metrics}

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create pipeline with distributed execution enabled
    pipeline = Pipeline(
        name="simple_classification",
        description="A simple classification pipeline with evaluation",
        distributed=True,  # Enable distributed execution
        max_workers=2     # Use 2 worker threads
    )

    # Define steps
    load_step = Step(
        name="data_loading",
        fn=data_loading,
        description="Load and split sample data"
    )

    preprocess_step = Step(
        name="preprocessing",
        fn=preprocessing,
        description="Preprocess features"
    )

    train_step = Step(
        name="training",
        fn=training,
        description="Train model"
    )

    eval_step = Step(
        name="evaluation",
        fn=evaluation,
        description="Evaluate model performance"
    )

    # Add steps to pipeline
    pipeline.add_step(load_step)
    pipeline.add_step(preprocess_step, dependencies=[load_step])
    pipeline.add_step(train_step, dependencies=[preprocess_step])
    pipeline.add_step(eval_step, dependencies=[train_step, preprocess_step])

    # Run pipeline
    pipeline.run()

if __name__ == "__main__":
    main()