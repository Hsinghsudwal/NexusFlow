
import os
import pickle
from typing import Any, Optional

class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, store_path: str = "artifacts"):
        self.store_path = store_path
        os.makedirs(self.store_path, exist_ok=True)

    def save_artifact(self, artifact: Any, name: str) -> None:
        """Save an artifact to disk."""
        artifact_path = os.path.join(self.store_path, f"{name}.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Artifact '{name}' saved to {artifact_path}")

    def load_artifact(self, name: str) -> Optional[Any]:
        """Load an artifact from disk."""
        artifact_path = os.path.join(self.store_path, f"{name}.pkl")
        if os.path.exists(artifact_path):
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)
            print(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
        else:
            print(f"Artifact '{name}' not found in {artifact_path}")
            return None

    def list_artifacts(self) -> list:
        """
        List all artifacts in the store.
        Returns:
            list: List of artifact names.
        """
        artifacts = [
            f.replace(".pkl", "") for f in os.listdir(self.store_path) if f.endswith(".pkl")
        ]
        logger.info(f"Artifacts in store: {artifacts}")
        return artifacts


class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")



from concurrent.futures import ThreadPoolExecutor
from src.ml_customer_churn.data_ingestion import DataIngestion
from src.ml_customer_churn.data_validation import DataValidation
# from src.ml_customer_churn.data_transformation import DataTransformation
# from src.ml_customer_churn.model_trainer import ModelTrainer
# from src.ml_customer_churn.model_evaluation import ModelEvaluation


class Stack:
    """Integrated component stack"""

    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Initialize components
        self.dataingest = DataIngestion()
        self.datavalidate = DataValidation()
        # self.datatransformer = DataTransformation()
        # self.modeltrainer = ModelTrainer()
        # self.modelevaluation = ModelEvaluation()
        # self.deployment = ModelDeployment()
        # self.monitoring = ModelMonitoring()
        # self.retraining = ModelRetraining()



from src.core.oi import Stack
from utils.helper import Config

# from src.ml_customer_churn.data_ingestion import DataIngestion
# from src.ml_customer_churn.data_validation import DataValidation
# from src.ml_customer_churn.data_transformation import DataTransformation


import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TrainingPipeline:
    """Main ML pipeline orchestrator"""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml")
        self.artifact_store = ArtifactStore()


    def run(self):

        logging.info(f"Starting pipeline for data at {self.path}")

        # Initialize stack
        # option_run = self.config.get('execution', {}).get('option_run', 'local')
        # option_scale = self.config.get('execution', {}).get('option_scale', 'local')

        my_stack = Stack("ml_pipeline", max_workers=4)

        # Data ingestion
        logging.info(f"Task_1: data ingestion ")
        train_data, test_data = my_stack.dataingest.data_ingestion(
            self.path, self.config
        )
        # print("train", train_data.head())

        logging.info(f"Task_2: data validation ")
       
        return "ytest"

class TrainingPipeline:
    """Main ML pipeline orchestrator"""

    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file("config/config.yml")
        self.artifact_store = ArtifactStore()

    def run(self):
        logging.info(f"Starting pipeline for data at {self.path}")

        # Initialize stack
        my_stack = Stack("ml_pipeline", max_workers=4)

        # Check if train_data and test_data are already available in the artifact store
        train_data = self.artifact_store.load_artifact("train_data")
        test_data = self.artifact_store.load_artifact("test_data")

        if train_data is None or test_data is None:
            # Data ingestion
            logging.info(f"Task_1: data ingestion")
            train_data, test_data = my_stack.dataingest.data_ingestion(
                self.path, self.config
            )
            # Save artifacts for future use
            self.artifact_store.save_artifact(train_data, "train_data")
            self.artifact_store.save_artifact(test_data, "test_data")
        else:
            logging.info("Skipping data ingestion. Loaded artifacts from store.")

        # Feature engineering
        logging.info(f"Task_3: feature engineering")
        feature_engineered_data = self.artifact_store.load_artifact("feature_engineered_data")

        if feature_engineered_data is None:
            # Perform feature engineering
            # feature_engineered_data = my_stack.feature_engineer.transform(train_data, test_data)
            # Save the feature-engineered data
            self.artifact_store.save_artifact(feature_engineered_data, "feature_engineered_data")
        else:
            logging.info("Skipping feature engineering. Loaded artifacts from store.")

        # Model training
        logging.info(f"Task_4: model training")
        trained_model = self.artifact_store.load_artifact("trained_model")

        if trained_model is None:
            # Train the model
            # trained_model = my_stack.modeltrainer.train(feature_engineered_data)
            # Save the trained model
            self.artifact_store.save_artifact(trained_model, "trained_model")
        else:
            logging.info("Skipping model training. Loaded artifacts from store.")

        return "Pipeline execution completed."



class DataIngestion:
    """Data ingestion component"""

    @task
    def data_ingestion(
        self, path: str, config: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into training and test sets"""
        logger.info(f"Loading data from {path}")

        try:
            data = pd.read_csv(path)

            # Split data according to config
            train_size = config.get("data_split", {}).get("train_size", 0.8)
            target_column = config.get("data", {}).get("target_column")

            if not target_column:
                raise ValueError("Target column not specified in config")

            X = data.drop(columns=[target_column])
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42
            )

            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            logger.info(
                f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}"
            )

            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise


from pipelines.training_pipeline import TrainingPipeline

if __name__ == "__main__":

    path = "data/churn-train.csv"
    pipeline = TrainingPipeline(path)
    # results = pipeline.run()
    # print(f"Pipeline Results: {results}")
    if pipeline.run() is not None:
        print("Pipeline executed successfully!")
        # Uncomment to check retraining status
        # retraining_status = pipeline.pipeline_stack.get_artifact("retraining_decision")
        # if retraining_status and retraining_status.data.get('required'):
        #     print("System requires retraining based on:")
        #     print(json.dumps(retraining_status.data, indent=2))
    else:
        print("Pipeline execution failed")



