# 15. Example of building a simple pipeline
def example_pipeline():
    """Example of how to build a pipeline."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Define pipeline steps as functions
    def load_data(data_path: str) -> pd.DataFrame:
        """Load raw data."""
        return pd.read_csv(data_path)
    
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        # Drop missing values
        df = df.dropna()
        
        # Convert categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
            
        return df
    
    def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets."""
        target = 'target'  # Assuming 'target' is the target column
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
        """Train a random forest model."""
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(model, X_test, y_test):
        """Evaluate the model."""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}
    
    # Create nodes
    load_node = node_from_func(
        load_data,
        inputs="data_path",
        outputs="raw_data",
        name="load_data",
        tags={"data", "ingestion"}
    )
    
    preprocess_node = node_from_func(
        preprocess_data,
        inputs="raw_data",
        outputs="processed_data",
        name="preprocess_data",
        tags={"data", "preprocessing"}
    )
    
    split_node = node_from_func(
        split_data,
        inputs={"df": "processed_data", "test_size": "params:test_size"},
        outputs=["X_train", "X_test", "y_train", "y_test"],
        name="split_data",
        tags={"data", "splitting"}
    )
    
    train_node = node_from_func(
        train_model,
        inputs={
            "X_train": "X_train",
            "y_train": "y_train",
            "n_estimators": "params:n_estimators"
        },
        outputs="model",
        name="train_model",
        tags={"model", "training"}
    )
    
    evaluate_node = node_from_func(
        evaluate_model,
        inputs={"model": "model", "X_test": "X_test", "y_test": "y_test"},
        outputs="metrics",
        name="evaluate_model",
        tags={"model", "evaluation"}
    )
    
    # Create pipeline
    pipeline = pipeline_from_nodes(
        load_node,
        preprocess_node,
        split_node,
        train_node,
        evaluate_node,
        name="example_pipeline"
    )
    
    return pipeline