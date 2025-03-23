# example.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from nexusml.core.metadata import MetadataStore
from nexusml.pipelines.pipeline import Pipeline
from nexusml.pipelines.executor import PipelineExecutor
from nexusml.steps.base_step import BaseStep
from nexusml.steps.decorators import step
from nexusml.core.context import ExecutionContext

# Step 1: Data Loading
@step(name="load_data", description="Load the Iris dataset")
def load_data(context: ExecutionContext) -> dict:
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    
    return {
        "X": X,
        "y": y,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names
    }

# Step 2: Data Preprocessing
@step(name="preprocess_data", description="Split the data into train and test sets")
def preprocess_data(context: ExecutionContext, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return {
        "X_