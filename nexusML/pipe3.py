# core/pipeline.py
import os
import dvc.api
from feast import FeatureStore
from shap import Explainer, force_plot
from kubernetes.client.rest import ApiException

# ... Previous imports remain same ...

# ---------- Feature Store Integration ----------
class FeatureStoreClient:
    def __init__(self, config: Dict):
        self.store = FeatureStore(repo_path=config['feature_store']['repo_path'])
        
    def get_features(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        return self.store.get_historical_features(
            entity_df=entity_df,
            features=config['feature_store']['feature_views']
        ).to_df()

# ---------- DVC Integration ----------
class DVCClient:
    def __init__(self, config: Dict):
        self.config = config
        
    def version_data(self, path: str):
        with dvc.api.open(path, repo=self.config['dvc']['repo_url']) as fd:
            return fd.read()

# ---------- Shadow Deployment ----------
class ShadowDeployment:
    def __init__(self, main_model, shadow_model):
        self.main_model = main_model
        self.shadow_model = shadow_model
        
    def predict(self, data):
        main_pred = self.main_model.predict(data)
        shadow_pred = self.shadow_model.predict(data)
        
        # Log discrepancies
        if main_pred != shadow_pred:
            logging.warning(f"Prediction mismatch: {main_pred} vs {shadow_pred}")
        return main_pred

# ---------- A/B Testing ----------
class ABTestRouter:
    def __init__(self, config: Dict):
        self.split_ratio = config['ab_testing']['split_ratio']
        
    def route(self, request) -> str:
        if 'X-Model-Variant' in request.headers:
            return request.headers['X-Model-Variant']
        return 'A' if random.random() < self.split_ratio else 'B'

# ---------- Explainability Dashboard ----------
class ExplainabilityDashboard:
    def __init__(self, config: Dict):
        self.explainer = Explainer(config['model'])
        self.dashboard_path = config['explainability']['dashboard_path']
        
    def generate_shap_report(self, data: pd.DataFrame):
        shap_values = self.explainer.shap_values(data)
        force_plot(self.explainer.expected_value, shap_values, data)
        plt.savefig(os.path.join(self.dashboard_path, 'shap_plot.html'))

# ---------- Zero Downtime Deployment ----------
class ZeroDowntimeDeployer:
    def __init__(self, config: Dict):
        self.config = config
        self.k8s_orchestrator = KubernetesOrchestrator(config)
        
    def blue_green_deploy(self, new_version: str):
        try:
            # Create new service
            self.k8s_orchestrator.deploy_service(
                self._create_deployment_spec(new_version)
            )
            
            # Update ingress
            self._switch_ingress_traffic(new_version)
            
            # Remove old deployment
            self._cleanup_previous_versions()
            return True
        except ApiException as e:
            logging.error(f"Blue/green deployment failed: {e}")
            return False

# ---------- Enhanced Data Ingestion ----------
class DataIngestion:
    def __init__(self, config: Dict):
        self.fs = FeatureStoreClient(config)
        self.dvc = DVCClient(config)
        
    def load_data(self):
        with self.dvc.version_data(self.config['data_path']) as data:
            entity_df = pd.read_csv(data)
            return self.fs.get_features(entity_df)

# ---------- Updated Deployment Class ----------
class Deployment:
    def __init__(self, config: Dict):
        # ... Previous initialization ...
        self.ab_router = ABTestRouter(config)
        self.shadow_model = None
        self.explain_dashboard = ExplainabilityDashboard(config)
        
        if config['shadow_deployment']['enabled']:
            self.shadow_model = load_model(config['shadow_deployment']['model_path'])
            
    @self.app.route('/explain', methods=['POST'])
    @auth.login_required
    def explain_prediction():
        data = request.json
        self.explain_dashboard.generate_shap_report(data)
        return send_file(os.path.join(config['explainability']['dashboard_path'], 'shap_plot.html'))

    def predict(self):
        @self.app.route('/predict', methods=['POST'])
        @self.auth.login_required
        def prediction():
            # A/B Testing
            model_variant = self.ab_router.route(request)
            model = self.model_registry.get_model(model_variant)
            
            # Shadow Deployment
            if self.shadow_model:
                shadow_pred = self.shadow_model.predict(request.json)
                self.monitoring.track_shadow_prediction(shadow_pred)
                
            # Main prediction
            pred = model.predict(request.json)
            return jsonify({"prediction": pred, "model_version": model_variant})

# ---------- Updated Training Pipeline ----------
class TrainingPipeline:
    def __init__(self, config: Dict):
        # ... Previous initialization ...
        self.feature_store = FeatureStoreClient(config)
        self.dvc = DVCClient(config)
        self.deployer = ZeroDowntimeDeployer(config)
        
    @flow(name="retraining_workflow")
    def retrain(self):
        with dvc.api.open(self.config['data_path']) as data:
            df = pd.read_csv(data)
            features = self.feature_store.get_features(df)
            
            # Train new model version
            new_model = ModelTrainer(self.config).train(features)
            
            # Zero-downtime deployment
            if self.config['deployment']['zero_downtime']:
                self.deployer.blue_green_deploy(new_model.version)
                
            return new_model

# ---------- Configuration Additions ----------
"""
# config/config.yml
feature_store:
  repo_path: "feature_repo/"
  feature_views:
    - customer_features
    - transaction_features

dvc:
  repo_url: "https://github.com/your-org/data-repo"
  
ab_testing:
  split_ratio: 0.5
  variants:
    - A
    - B

shadow_deployment:
  enabled: true
  model_path: "models/shadow_model.pkl"

explainability:
  dashboard_path: "reports/explanations/"
  sample_size: 1000

zero_downtime:
  strategy: "blue_green"
  ingress_name: "model-ingress"
"""



# feature_repo/feature_store.yaml
project: mlops_feature_store
registry: data/registry.db
provider: local
online_store:
    type: redis
    connection_string: "redis:6379"

# Enable in config
shadow_deployment:
  enabled: true
  model_path: "models/shadow_model_v2.pkl"

# Create multiple model variants
kubectl apply -f ab-testing/service-a.yaml
kubectl apply -f ab-testing/service-b.yaml


# Blue-green deployment strategy
deployer.blue_green_deploy("v2.0.0")

# Initialize DVC
dvc init
dvc remote add -d storage s3://your-bucket/dvc-storage
dvc add data/raw
dvc push

# Generate SHAP explanations
dashboard = ExplainabilityDashboard(config)
dashboard.generate_shap_report(test_sample)


class EnhancedMonitoring(Monitoring):
    def __init__(self, config):
        super().__init__(config)
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.explainability_tracker = ExplainabilityTracker()
        
    def track_model_performance(self, y_true, y_pred):
        self.performance_monitor.update(y_true, y_pred)
        self._alert_on_perf_degradation()