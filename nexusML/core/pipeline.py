from concurrent.futures import ThreadPoolExecutor
from prefect import flow, task
from .data_ingestion import DataIngestion
from .model_training import ModelTrainer
from .deployment import Deployment
from .monitoring import Monitoring
from .retraining import Retraining

class TrainingPipeline:
    def __init__(self, config):
        self.config = config

    @flow(name="mlops_pipeline")
    def run(self):
        # Data Ingestion
        data = DataIngestion(self.config).load_data()
        
        # Model Training
        model = ModelTrainer(self.config).train(data)
        
        # Deployment
        deployment = Deployment(self.config)
        monitoring = Monitoring(self.config)
        retraining = Retraining(self.config)

        # Execute pipeline steps in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(deployment.deploy),
                executor.submit(monitoring.generate_report, data, data),
                executor.submit(retraining.check_retraining, {"drift_detected": False})
            ]
            return [f.result() for f in futures]