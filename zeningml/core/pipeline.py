from core.pipeline_versioning import PipelineVersioner
from core.metadata_tracking import PipelineMetadata

class Pipeline:
    def __init__(self, stack):
        self.stack = stack
        self.steps = []
        self.version_info = stack.versioner.get_version_info()
        self.metadata = PipelineMetadata(
            run_id=self.version_info["run_id"],
            execution_date=datetime.now(),
            parameters={},
            metrics={},
            artifacts={},
            git_commit=self.version_info["git_commit"]
        )

    def add_step(self, step):
        self.steps.append(step)

    def execute(self):
        # Execute pipeline with tracking
        try:
            self.stack.metadata_tracker.start_run()
            
            for step in self.steps:
                step_output = step.execute()
                
                # Capture artifacts
                for artifact_name, artifact_data in step_output.artifacts.items():
                    self.stack.artifact_store.save_artifact(
                        artifact_data.data,
                        artifact_data.subdir,
                        artifact_data.name
                    )
                    self.metadata.artifacts[artifact_name] = os.path.join(
                        self.stack.artifact_store.base_path,
                        artifact_data.subdir,
                        artifact_data.name
                    )
                
                # Capture parameters and metrics
                self.metadata.parameters.update(step_output.parameters)
                self.metadata.metrics.update(step_output.metrics)
            
            self.stack.metadata_tracker.capture_metadata(self.metadata)
        finally:
            self.stack.metadata_tracker.end_run()


class Pipeline:
    def execute(self):
        try:
            # CI/CD Pipeline Trigger
            if self.config.get('cicd.enabled'):
                self.stack.cicd.run_pipeline_tests()
                
            # Data Validation
            validation_result = self.stack.validator.validate(raw_data)
            if not validation_result['valid']:
                raise DataQualityError(validation_result['results'])
                
            # Feature Engineering
            features = self.stack.feature_store.get_historical_features(entity_data)
            
            # Model Training
            model = self.train_model(features)
            
            # Initialize Monitoring
            reference_data = features.sample(1000)
            self.stack.drift_detectors['data_drift'] = DataDriftDetector(reference_data)
            
            # CI/CD Deployment
            if self.config.get('deploy'):
                self.deploy_model(model)
                
        except Exception as e:
            self.stack.monitor.trigger_alert(f"Pipeline failed: {str(e)}")
            raise

class Pipeline:
    def execute(self):
        # Automated Hyperparameter Tuning
        optimizer = HyperparameterOptimizer(self.config)
        best_params = optimizer.optimize(
            objective=self.train_objective,
            search_space=self.search_space
        )
        
        # Model Training with Best Params
        model = self.train_model(best_params)
        
        # Model Explainability
        if "shap" in self.config.get("explainability.methods"):
            explainer = SHAPExplainer(model, self.training_data)
            explanation = explainer.explain_sample(self.test_sample)
            explainer.log_explanation(explanation)
        
        # Bandit Deployment Strategy
        deployment_strategy = MultiArmBanditStrategy(
            models=[model, self.production_model],
            initial_split=self.config.get("deployment.initial_split")
        )
        
        # Deploy with Strategy
        self.deploy(deployment_strategy.select_model())
        
        # Setup Alerts
        alert_manager = AlertManager(self.config)
        alert_manager.register_handler(SlackAlerter(self.config))
        alert_manager.register_handler(PagerDutyAlerter(self.config))



class MLOpsPipeline:
    def __init__(self, config):
        self.config = Config(config)
        self.initialize_components()
        
    def initialize_components(self):
        self.registry = ModelRegistry(self.config)
        self.rollback_manager = RollbackManager(self.registry)
        self.cloud_monitor = CloudWatchMetrics(self.config)
        self.data_quality = DataQualityMonitor(self.config)
        self.feature_validator = FeatureValidationPipeline(self.config)

    def execute(self):
        try:
            # Data Quality Check
            raw_data = self.load_data()
            self.data_quality.continuous_validation(raw_data)
            
            # Feature Validation
            features = self.feature_engineering(raw_data)
            feature_validation = self.feature_validator.validate_new_features(features)
            
            # Model Training
            model = self.train_model(features)
            versioned_model = VersionedModel(model, self.config)
            
            # Registry & Deployment
            self.registry.register_model(versioned_model)
            self.rollback_manager.deploy(versioned_model)
            
            # Cloud Monitoring
            self.cloud_monitor.put_metric("ModelAccuracy", versioned_model.metadata["accuracy"])
            
        except Exception as e:
            self.rollback_manager.rollback()
            raise

    def monitor_production(self):
        while True:
            # Real-time monitoring
            self.cloud_monitor.put_metric("Predictions/Min", self.monitor.get_throughput())
            self.data_quality.continuous_validation(self.get_production_data())
            
            # Auto-rollback check
            if AutoRollback(self.monitor, self.registry).check_and_rollback():
                self.cloud_monitor.put_metric("System/Rollbacks", 1)
            
            time.sleep(60)


class FullMLOpsPipeline:
    def __init__(self, config):
        self.config = Config(config)
        self.ab_testing = ABTestingFramework(config)
        self.cost_monitor = CloudCostMonitor(config)
        self.security = SecurityCompliance(config)
        self.serving_factory = ServerFactory()
        self.cloud_client = MultiCloudArtifactStore(config)

    def execute(self):
        # Model Training
        model = self.train_model()
        
        # Security Checks
        compliance = self.security.run_checks(data, model)
        if not self.pass_security(compliance):
            raise SecurityViolationError(compliance)
        
        # Multi-cloud Deployment
        self.cloud_client.save_model(model, "prod_models")
        
        # AB Testing Setup
        self.ab_testing.create_experiment(
            name="prod_experiment",
            variants=[model.current_version, model.previous_version],
            traffic_split=[0.8, 0.2]
        )
        
        # Cost-aware Serving
        server = self.serving_factory.create_server(model, self.config)
        server.start()
        
        # Continuous Monitoring
        while True:
            self.monitor_costs()
            self.adjust_scaling()
            self.check_compliance()
            time.sleep(3600)

    def monitor_costs(self):
        cost = self.cost_monitor.get_current_cost()
        if cost > self.config['cost_monitoring']['budget_alert']:
            self.alert_manager.trigger(f"Cost exceeded ${cost}")

    def adjust_scaling(self):
        current_load = self.monitor.get_throughput()
        desired_replicas = min(
            max(current_load // 100, self.config['min_replicas']),
            self.config['max_replicas']
        )
        self.server.scale(desired_replicas)

# core/step_pipeline.py
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

class Step:
    """Base class for pipeline steps"""
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.input_artifacts: Dict[str, str] = {}
        self.output_artifacts: Dict[str, str] = {}
        self._dependencies: List['Step'] = []

    def add_dependency(self, step: 'Step'):
        """Add another step that must complete before this one"""
        self._dependencies.append(step)

    def execute(self, context: 'PipelineContext') -> Dict[str, Any]:
        """To be implemented by concrete steps"""
        raise NotImplementedError

    @property
    def step_hash(self) -> str:
        """Unique hash for the step based on parameters and dependencies"""
        hash_data = {
            'name': self.name,
            'params': self.parameters,
            'deps': [dep.step_hash for dep in self._dependencies]
        }
        return hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

class PipelineContext:
    """Shared context for pipeline execution"""
    def __init__(self, config: 'Config', artifact_store: 'ArtifactStore'):
        self.config = config
        self.artifact_store = artifact_store
        self.metadata: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.execution_id = self._generate_execution_id()

    def _generate_execution_id(self) -> str:
        return f"pipeline_{self.start_time.strftime('%Y%m%d%H%M%S')}"

class StepPipeline:
    """Orchestrates execution of multiple steps with dependency management"""
    def __init__(self, name: str, config: 'Config'):
        self.name = name
        self.config = config
        self.steps: Dict[str, Step] = {}
        self._execution_order: List[Step] = []
        self.artifact_store = ArtifactStore(config)
        self.context = PipelineContext(config, self.artifact_store)

    def add_step(self, step: Step):
        """Register a step in the pipeline"""
        if step.name in self.steps:
            raise ValueError(f"Step {step.name} already exists in pipeline")
        self.steps[step.name] = step

    def resolve_dependencies(self):
        """Topological sort to determine execution order"""
        visited = set()
        order = []

        def visit(step):
            if step.name not in visited:
                visited.add(step.name)
                for dep in step._dependencies:
                    visit(dep)
                order.append(step)

        for step in self.steps.values():
            visit(step)

        self._execution_order = order

    def run(self, orchestrator: Optional[str] = None):
        """Execute the pipeline with optional orchestrator integration"""
        self.resolve_dependencies()
        execution_plan = self._create_execution_plan()

        if orchestrator:
            self._delegate_to_orchestrator(orchestrator, execution_plan)
        else:
            self._execute_locally(execution_plan)

    def _create_execution_plan(self) -> List[Dict]:
        """Generate execution plan with caching information"""
        plan = []
        for step in self._execution_order:
            cache_key = self._generate_cache_key(step)
            artifact_path = f"{self.context.execution_id}/{step.name}"
            
            plan.append({
                'step': step,
                'cache_key': cache_key,
                'artifact_path': artifact_path,
                'cached': self.artifact_store.exists(cache_key)
            })
        return plan

    def _generate_cache_key(self, step: Step) -> str:
        """Generate unique cache key for step results"""
        return f"{step.name}_{step.step_hash}"

    def _execute_locally(self, execution_plan: List[Dict]):
        """Local execution without external orchestrator"""
        logging.info(f"Starting pipeline '{self.name}' with {len(execution_plan)} steps")
        
        for step_info in execution_plan:
            step = step_info['step']
            if step_info['cached']:
                logging.info(f"Skipping cached step: {step.name}")
                continue

            logging.info(f"Executing step: {step.name}")
            try:
                result = step.execute(self.context)
                self._store_step_results(step, result, step_info['artifact_path'])
                self._update_metadata(step, result)
            except Exception as e:
                logging.error(f"Step {step.name} failed: {str(e)}")
                raise PipelineExecutionError(f"Step {step.name} failed") from e

        logging.info(f"Pipeline '{self.name}' completed successfully")

    def _store_step_results(self, step: Step, result: Dict, artifact_path: str):
        """Store artifacts and update context"""
        for artifact_name, artifact_data in result.get('artifacts', {}).items():
            self.artifact_store.save_artifact(
                artifact_data,
                artifact_path,
                f"{artifact_name}.pkl"
            )
            step.output_artifacts[artifact_name] = f"{artifact_path}/{artifact_name}.pkl"

    def _update_metadata(self, step: Step, result: Dict):
        """Update pipeline metadata store"""
        self.context.metadata[step.name] = {
            'parameters': step.parameters,
            'metrics': result.get('metrics', {}),
            'artifacts': step.output_artifacts,
            'execution_time': datetime.now() - self.context.start_time
        }

    def _delegate_to_orchestrator(self, orchestrator: str, execution_plan: List[Dict]):
        """Generate orchestrator-specific workflow definitions"""
        if orchestrator == "prefect":
            from orchestration.prefect_orchestrator import create_prefect_flow
            create_prefect_flow(execution_plan, self.context)
        elif orchestrator == "airflow":
            from orchestration.airflow_orchestrator import create_airflow_dag
            create_airflow_dag(execution_plan, self.context)
        elif orchestrator == "kubeflow":
            from orchestration.kubeflow_orchestrator import create_kubeflow_pipeline
            create_kubeflow_pipeline(execution_plan, self.context)
        else:
            raise ValueError(f"Unsupported orchestrator: {orchestrator}")

class PipelineExecutionError(Exception):
    """Custom exception for pipeline failures"""
    pass
            

