from nexusml.pipeline import Pipeline
from nexusml.step import Step
from nexusml.runner import Runner
from nexusml.logger import Logger
from nexusml.tracker import Tracker
from nexusml.artifact import Artifact
from nexusml.artifact_store import ArtifactStore
from nexusml.orchestrator import Orchestrator

# Step 1: Define the pipeline
pipeline = Pipeline(name="MLPipeline")

# Step 2: Define pipeline steps
def preprocess_data():
    print("Preprocessing data...")

def train_model():
    print("Training model...")

pipeline.add_step(Step("Preprocessing", preprocess_data))
pipeline.add_step(Step("Model Training", train_model))

# Step 3: Run the pipeline
runner = Runner(pipeline)
runner.start()

# Step 4: Track and log experiments
logger = Logger()
logger.log("Experiment started.")

tracker = Tracker()
tracker.log_metric("accuracy", 0.95)

# Step 5: Manage artifacts
artifact = Artifact(name="model_v1", data={"model_data": "binary_data"})
artifact.save()

artifact_store = ArtifactStore()
artifact_store.add(artifact)






import logging
from typing import Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nexusml import Pipeline, Step
from nexusml.core.metrics import ModelEvaluator
from nexusml.storage.cloud import S3StorageProvider
from nexusml.artifact_store import ArtifactStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Define the pipeline steps
def data_loading(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Load and split data."""
    logger.info("Loading and splitting data...")

    data_path = config.get("data_path", "../data/data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    X_train_normalized = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test_normalized = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

    return {
        "X_train_processed": X_train_normalized,
        "X_test_processed": X_test_normalized,
        "y_train": artifacts["y_train"],
        "y_test": artifacts["y_test"]
    }

def training(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Train a simple model."""
    logger.info("Training model...")
    model = LogisticRegression()
    X_train = artifacts["X_train_processed"]
    y_train = artifacts["y_train"]
    model.fit(X_train, y_train)
    return {"model": model}

def evaluation(artifacts: Dict, config: Dict[str, Any]) -> Dict:
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    model = artifacts["model"]
    X_test = artifacts["X_test_processed"]
    y_test = artifacts["y_test"]

    y_pred = model.predict(X_test)
    metrics = ModelEvaluator.evaluate_classifier(y_test, y_pred)
    logger.info(f"Model metrics: {metrics}")
    return {"metrics": metrics}

# Step 2: Define the pipeline
def main():
    # Define configuration with the data path
    config = {
        "data_path": "../data/data.csv"  # Adjust this to the actual data location
    }

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
    pipeline.run(config=config)  # Pass config with the data path

if __name__ == "__main__":
    main()
