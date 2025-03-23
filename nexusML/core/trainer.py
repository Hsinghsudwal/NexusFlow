import mlflow
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Train model
        model = RandomForestClassifier(**self.config['model']['hyperparameters'])
        model.fit(X, y)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_params(self.config['model']['hyperparameters'])
            mlflow.sklearn.log_model(model, "model")
        
        return model