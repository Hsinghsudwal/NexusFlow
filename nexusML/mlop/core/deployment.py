from flask import Flask, jsonify
from kubernetes import client, config
from prometheus_client import start_http_server, Counter, Summary
from prefect import task
import logging

logger = logging.getLogger(__name__)

class ModelDeployment:
    def __init__(self):
        self.app = Flask(__name__)
        self.model = None
        self.config = None

    @task
    def deploy_model(self, model: Any, config: Dict) -> bool:
        """Deploy model using Flask and Kubernetes."""
        logger.info("Starting model deployment")

        try:
            self.model = model
            self.config = config

            # Start Prometheus metrics server
            start_http_server(config['monitoring']['prometheus_port'])

            # Deploy to Kubernetes if enabled
            if config['deployment']['kubernetes']['enabled']:
                self.deploy_kubernetes()

            # Start Flask server
            self.app.run(host='0.0.0.0', port=config['deployment']['flask_port'])
            return True

        except Exception as e:
            logger.error(f"Error during model deployment: {e}")
            return False

    def deploy_kubernetes(self):
        """Deploy model to Kubernetes."""
        config.load_kube_config()
        apps_v1 = client.AppsV1Api()

        # Apply deployment and ingress
        with open("kubernetes/deployment.yaml") as f:
            dep = yaml.safe_load(f)
            apps_v1.create_namespaced_deployment(
                body=dep,
                namespace=self.config['deployment']['kubernetes']['namespace']
            )

        with open("kubernetes/ingress.yaml") as f:
            ingress = yaml.safe_load(f)
            networking_v1 = client.NetworkingV1Api()
            networking_v1.create_namespaced_ingress(
                body=ingress,
                namespace=self.config['deployment']['kubernetes']['namespace']
            )