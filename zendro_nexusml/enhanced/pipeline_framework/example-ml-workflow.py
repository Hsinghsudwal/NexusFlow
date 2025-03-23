# example_ml_workflow.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

from datapipe import node, Pipeline, ProjectContext

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define data processing nodes
@node(outputs="raw_data")
def load_data():
    """Load raw data from source."""
    # For example purposes, we're creating synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features (age, income, education_level, etc.)
    X = np.random.randn(n_samples, 5)
    
    # Create target (customer churn: 0 or 1)
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    
    # Create dataframe
    df = pd.DataFrame(
        X, 
        columns=['age', 'income', 'education', 'tenure', 'products_owned']
    )
    df['churn'] = y
    
    return df

@node(inputs="raw_data", outputs=["X_data", "y_data"])
def preprocess_data(raw_data):
    """Extract features and target, handle missing values."""
    # Handle any missing values
    data_clean = raw_data.fillna(0)
    
    # Split into features and target
    X = data_clean.drop('churn', axis=1)
    y = data_clean['churn']
    
    return X, y

@node(inputs=["X_data", "y_data"], outputs=["X_train", "X_test", "y_train", "y_test"])
def split_data(X_data, y_data):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

@node(inputs=["X_train", "X_test"], outputs=["X_train_scaled", "X_test_scaled", "scaler"])
def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled, scaler

@node(inputs=["X_train_scaled", "y_train"], outputs="model")
def train_model(X_train_scaled, y_train):
    """Train a RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

@node(inputs=["model", "X_test_scaled", "y_test"], outputs=["predictions", "metrics"])
def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    return predictions, metrics

@node(inputs=["model", "X_data"], outputs="feature_importance")
def analyze_features(model, X_data):
    """Analyze feature importance."""
    feature_importance = pd.DataFrame({
        'feature': X_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(
    nodes=[
        load_data,
        preprocess_data,
        split_data,
        scale_features
    ],
    name="preprocessing_pipeline"
)

# Create the modeling pipeline
modeling_pipeline = Pipeline(
    nodes=[
        train_model,
        evaluate_model,
        analyze_features
    ],
    name="modeling_pipeline"
)

# Create the full pipeline
full_pipeline = Pipeline(
    nodes=[
        load_data,
        preprocess_data,
        split_data,
        scale_features,
        train_model,
        evaluate_model,
        analyze_features
    ],
    name="full_ml_pipeline"
)

if __name__ == "__main__":
    # Create project context
    context = ProjectContext("ml_project")
    
    # Run the full pipeline
    results = context.run_pipeline(full_pipeline)
    
    # Print results
    print(f"Model Accuracy: {results['metrics']['accuracy']:.4f}")
    print("\nFeature Importance:")
    print(results['feature_importance'])
