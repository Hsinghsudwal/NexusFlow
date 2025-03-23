# example_project/pipelines/data_engineering.py
from mlops_framework.core.pipeline import node, Pipeline
import pandas as pd
import numpy as np

@node(outputs=["cleaned_data"])
def load_and_clean_data(raw_data_path):
    """Load and clean the dataset."""
    # Load data
    df = pd.read_csv(raw_data_path)
    
    # Basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    return df

@node(inputs=["cleaned_data"], outputs=["train_data", "test_data"])
def split_data(cleaned_data, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    # Simple split based on random sampling
    mask = np.random.rand(len(cleaned_data)) < (1 - test_size)
    train_data = cleaned_data[mask]
    test_data = cleaned_data[~mask]
    
    return train_data, test_data

@node(inputs=["train_data"], outputs=["features"])
def create_features(train_data):
    """Create features from training data."""
    # Feature engineering
    features = train_data.copy()
    
    # Example feature: day of week from date column
    if 'date' in features.columns:
        features['day_of_week'] = pd.to_datetime(features['date']).dt.dayofweek
    
    # Example feature: binning a numeric column
    if 'age' in features.columns:
        features['age_group'] = pd.cut(features['age'], bins=[0, 18, 35, 50, 65, 100], 
                                    labels=['<18', '18-34', '35-49', '50-64', '65+'])
    
    return features

# Create the data engineering pipeline
data_engineering_pipeline = Pipeline(
    nodes=[
        load_and_clean_data,
        split_data,
        create_features
    ],
    inputs=["raw_data_path"],
    outputs=["features", "test_data"],
    name="data_engineering_pipeline"
)

# example_project/pipelines/data_science.py
from mlops_framework.core.pipeline import node, Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@node(inputs=["features"], outputs=["X_train", "y_train"])
def prepare_training_data(features, target_column="target"):
    """Prepare data for model training."""
    # Separate features and target
    X_train = features.drop(columns=[target_column])
    y_train = features[target_column]
    
    return X_train, y_train

@node(inputs=["X_train", "y_train