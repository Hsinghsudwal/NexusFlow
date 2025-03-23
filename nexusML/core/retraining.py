import dvc.api
import mlflow

class Retraining:
    def __init__(self, config):
        self.config = config

    def check_retraining(self, metrics):
        if metrics.get('drift_detected', False):
            self.retrain_model()

    def retrain_model(self):
        with dvc.api.open(self.config['data']['path']) as fd:
            data = pd.read_csv(fd)
            model = ModelTrainer(self.config).train(data)
            mlflow.register_model(model, self.config['model']['name'])