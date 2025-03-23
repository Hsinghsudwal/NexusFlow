from prefect import flow
from .data_ingestion import DataIngestion
from .feature_engineering import FeatureEngineering
from .model_training import ModelTraining
from .deployment import ModelDeployment
from .monitoring import ModelMonitoring

class TrainingPipeline:
    @flow(name="MLOps Pipeline")
    def run(self, config: Dict) -> Dict:
        """Run the complete MLOps pipeline."""
        data_ingestion = DataIngestion()
        feature_engineering = FeatureEngineering()
        model_training = ModelTraining()
        deployment = ModelDeployment()
        monitoring = ModelMonitoring()

        # Execute pipeline
        train_data, test_data = data_ingestion.data_ingestion(config['data']['path'], config)
        processed_data = feature_engineering.process_features(train_data, config)
        model = model_training.train_model(processed_data, config)
        deployment_success = deployment.deploy_model(model, config)
        dashboard_path = monitoring.setup_monitoring(test_data, config)

        return {
            'deployment': deployment_success,
            'monitoring': dashboard_path
        }