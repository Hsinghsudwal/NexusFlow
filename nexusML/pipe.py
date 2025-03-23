import logging
import yaml
import psycopg2
import boto3
import mlflow
from prometheus_client import start_http_server, Gauge
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_framework')

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


# Base class for all pipeline steps
class PipelineStep:
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Each step must implement the 'run' method.")


# Pipeline Orchestrator
class MLPipeline:
    def __init__(self, steps: List[PipelineStep], max_workers: int = 4):
        self.steps = steps
        self.max_workers = max_workers

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for step in self.steps:
                futures.append(executor.submit(step.run, data))

            for future in futures:
                try:
                    result = future.result()
                    data.update(result)
                except Exception as e:
                    logger.error(f"Error executing step: {e}")
                    raise

        return data

# Example steps
# Step 1: Data Ingestion
class DataIngestionStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Ingesting data...")
        # Simulate data ingestion
        data["raw_data"] = "raw_data"
        return {"raw_data": data["raw_data"]}

class DefaultDataIngestion(DataIngestion):
    def ingest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Ingesting data...")
        # Example: Fetch data from a database
        data = {"raw_data": "sample_data"}
        return data

# Step 2: Data Preprocessing
class DataPreprocessingStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Preprocessing data...")
        # Simulate data preprocessing
        data["processed_data"] = "processed_data"
        return {"processed_data": data["processed_data"]}

class DefaultDataPreprocessing(DataPreprocessing):
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Preprocessing data...")
        data["processed_data"] = "processed_data"
        return data


# Step 3: Model Training
class ModelTrainingStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Training model...")
        # Simulate model training
        data["model"] = "trained_model"
        return {"model": data["model"]}

# Step 4: Model Evaluation
class ModelEvaluationStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Evaluating model...")
        # Simulate model evaluation
        data["evaluation_metrics"] = {"accuracy": 0.95}
        return {"evaluation_metrics": data["evaluation_metrics"]}

# Step 5: Model Deployment
class ModelDeploymentStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Deploying model...")
        # Simulate model deployment
        data["deployment_status"] = "success"
        return {"deployment_status": data["deployment_status"]}

# Step 6: Monitoring
class MonitoringStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Monitoring model...")
        # Simulate monitoring
        data["monitoring_status"] = "stable"
        return {"monitoring_status": data["monitoring_status"]}

 from prometheus_client import Counter, Histogram

class MonitoringStep(PipelineStep):
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Monitoring model...")
        try:
            # Start Prometheus metrics server
            start_http_server(config["monitoring"]["prometheus_port"])

            # Define additional metrics
            latency_histogram = Histogram("model_latency_seconds", "Latency of model predictions")
            throughput_counter = Counter("model_throughput_total", "Total number of predictions")

            # Simulate metrics
            latency_histogram.observe(0.2)  # Example latency
            throughput_counter.inc(100)     # Example throughput

            # Simulate sending data to Grafana
            logger.info(f"Metrics sent to Grafana dashboard: {config['monitoring']['grafana_dashboard_url']}")
            data["monitoring_status"] = "stable"
            return {"monitoring_status": data["monitoring_status"]}
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            raise   

# Define the pipeline steps
# Main Execution
if __name__ == "__main__":
    # Define the pipeline steps
    steps = [
        DataIngestionStep(),
        DataPreprocessingStep(),
        ModelTrainingStep(),
        ModelEvaluationStep(),
        ModelDeploymentStep(),
        MonitoringStep()
    ]

    # Initialize and run the pipeline
    pipeline = MLPipeline(steps, max_workers=4)
    initial_data = {}
    final_data = pipeline.run(initial_data)

    logger.info(f"Pipeline execution completed. Final data: {final_data}")


class MLPipeline:
    def __init__(
        self,
        data_ingestion: DataIngestion,
        data_preprocessing: DataPreprocessing,
        model_training: ModelTraining,
        model_evaluation: ModelEvaluation,
        model_deployment: ModelDeployment,
        monitoring: Monitoring,
    ):
        self.data_ingestion = data_ingestion
        self.data_preprocessing = data_preprocessing
        self.model_training = model_training
        self.model_evaluation = model_evaluation
        self.model_deployment = model_deployment
        self.monitoring = monitoring

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        data = self.data_ingestion.ingest(config)
        data = self.data_preprocessing.preprocess(data)
        data = self.model_training.train(data)
        data = self.model_evaluation.evaluate(data)
        data = self.model_deployment.deploy(data)
        data = self.monitoring.monitor(data)
        return data

docker run -d --name prometheus -p 9090:9090 prom/prometheus
docker run -d --name grafana -p 3000:3000 grafana/grafana

FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "pipeline.py"]

docker build -t mlops-pipeline .
docker run -d --name mlops-pipeline mlops-pipeline


#kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-pipeline
  template:
    metadata:
      labels:
        app: mlops-pipeline
    spec:
      containers:
      - name: mlops-pipeline
        image: mlops-pipeline
        ports:
        - containerPort: 9090  # Prometheus metrics port

kubectl apply -f deployment.yaml
kubectl scale deployment mlops-pipeline --replicas=5


class ComponentFactory:
    @staticmethod
    def create_data_ingestion(config: Dict[str, Any]) -> DataIngestion:
        if config["type"] == "default":
            return DefaultDataIngestion()
        elif config["type"] == "s3":
            return S3DataIngestion(config["params"])
        elif config["type"] == "database":
            return DatabaseDataIngestion(config["params"])
        else:
            raise ValueError(f"Unknown data ingestion type: {config['type']}")

    @staticmethod
    def create_model_training(config: Dict[str, Any]) -> ModelTraining:
        if config["type"] == "default":
            return DefaultModelTraining()
        elif config["type"] == "mlflow":
            return MLflowModelTraining(config["params"])
        else:
            raise ValueError(f"Unknown model training type: {config['type']}")

    # Add similar methods for other components


from prometheus_client import start_http_server, Histogram, Counter

class PrometheusMonitoring(Monitoring):
    def __init__(self, config: Dict[str, Any]):
        self.port = config["port"]
        self.latency_histogram = Histogram("model_latency_seconds", "Latency of model predictions")
        self.throughput_counter = Counter("model_throughput_total", "Total number of predictions")

    def monitor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        start_http_server(self.port)
        self.latency_histogram.observe(0.2)  # Example latency
        self.throughput_counter.inc(100)     # Example throughput
        data["monitoring_status"] = "stable"
        return data

from tenacity import retry, stop_after_attempt, wait_fixed

class DatabaseDataIngestion(DataIngestion):
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def ingest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Ingesting data from database...")
        # Connect to database and fetch data
        return {"raw_data": "database_data"}

#running
if __name__ == "__main__":
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    pipeline = MLPipeline(
        data_ingestion=ComponentFactory.create_data_ingestion(config["data_ingestion"]),
        data_preprocessing=DefaultDataPreprocessing(),
        model_training=ComponentFactory.create_model_training(config["model_training"]),
        model_evaluation=DefaultModelEvaluation(),
        model_deployment=DefaultModelDeployment(),
        monitoring=PrometheusMonitoring(config["monitoring"]),
    )

    result = pipeline.run(config)
    logger.info(f"Pipeline execution completed. Final data: {result}")



    import yaml
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import mlflow
from prefect import flow, task
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import prometheus_client
from prometheus_client import start_http_server

class Config:
    @staticmethod
    def load_file(config_path: str) -> Dict[str, Any]:
        with open(config_path) as f:
            return yaml.safe_load(f)

class DataIngestion:
    @staticmethod
    def data_ingestion(path: str, config: Dict) -> tuple:
        df = pd.read_csv(path)
        train_df, test_df = train_test_split(
            df,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        return train_df, test_df

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None

    def train(self, train_data: pd.DataFrame):
        # Add your actual model training logic here
        # Example:
        # self.model = RandomForestClassifier(**self.config['model'])
        # self.model.fit(train_data.drop('target', axis=1), train_data['target'])
        
        # Log experiment with MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        with mlflow.start_run():
            mlflow.log_params(self.config['model'])
            # mlflow.log_metrics(metrics)
            # mlflow.sklearn.log_model(self.model, "model")
        
        return self.model

class Deployment:
    def __init__(self, config: Dict):
        self.config = config
        self.app = Flask(__name__)

    def predict(self):
        @self.app.route('/predict', methods=['POST'])
        def prediction():
            data = request.json
            # prediction = self.model.predict(data)
            return jsonify({"prediction": 0})  # Replace with actual prediction

    def deploy(self):
        self.predict()
        self.app.run(host=self.config['deployment']['host'],
                     port=self.config['deployment']['port'])
        return True

class Monitoring:
    def __init__(self, config: Dict):
        self.config = config
        self.report = Report(metrics=[DatasetDriftMetric()])

    def generate_report(self, reference_data, current_data):
        self.report.run(reference_data=reference_data, current_data=current_data)
        self.report.save_html(self.config['monitoring']['report_path'])
        return {"dashboard": "http://localhost:3000/dashboard"}  # Grafana URL

class Retraining:
    def __init__(self, config: Dict):
        self.config = config

    def check_retraining(self, metrics: Dict) -> bool:
        # Add retraining logic based on metrics
        return metrics.get('drift_detected', False)

class Stack:
    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append(self.executor.submit(fn, *args, **kwargs))

    def run(self) -> Dict:
        results = {}
        for future in self.tasks:
            task_name, result = future.result()
            results[task_name] = result
        return results

class TrainingPipeline:
    def __init__(self, path: str):
        self.path = path
        self.config = Config.load_file('config/config.yml')
        self.model = None
        self.metrics = {}

    @flow(name="training_workflow")
    def run(self):
        # Data ingestion
        train_data, test_data = DataIngestion().data_ingestion(self.path, self.config)
        
        # Model training
        self.model = ModelTrainer(self.config).train(train_data)
        
        # Initialize components
        deployment = Deployment(self.config)
        monitoring = Monitoring(self.config)
        retraining = Retraining(self.config)

        # Setup stack
        my_stack = Stack("mlops_pipeline", max_workers=3)
        
        # Add tasks to stack
        my_stack.add_task(lambda: ('deployment', deployment.deploy()))
        my_stack.add_task(lambda: ('monitoring', monitoring.generate_report(train_data, test_data)))
        my_stack.add_task(lambda: ('retraining', retraining.check_retraining(self.metrics)))

        # Run pipeline
        return my_stack.run()

def main():
    pipeline = TrainingPipeline(path='data/data.csv')
    results = pipeline.run()
    print("Pipeline execution results:")
    print(results)

if __name__ == '__main__':
    # Start monitoring server
    start_http_server(8000)
    main()

