import mlflow
from sklearn.ensemble import RandomForestClassifier
from prefect import task
import logging

logger = logging.getLogger(__name__)

class ModelTraining:
    @task
    def train_model(self, train_data: pd.DataFrame, config: Dict) -> Any:
        """Train machine learning model."""
        logger.info("Starting model training")

        try:
            target_column = config['data']['target_column']
            X = train_data.drop(columns=[target_column])
            y = train_data[target_column]

            model = RandomForestClassifier(
                n_estimators=config['model']['n_estimators'],
                max_depth=config['model']['max_depth'],
                random_state=42
            )

            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(config['model'])
                model.fit(X, y)
                mlflow.sklearn.log_model(model, "model")

            logger.info("Model training complete")
            return model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise