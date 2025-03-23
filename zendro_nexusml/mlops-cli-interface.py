# cli.py
import argparse
import json
import os
import sys
from typing import Dict, List, Any

from core.pipeline import Pipeline, PipelineStep, Artifact
from core.model_registry import ModelRegistry
from core.deployment import Deployment
from core.monitoring import ModelMonitor
from core.config import Config

class CLI:
    def __init__(self):
        self.config = Config()
        self.parser = self.create_parser()
        
    def create_parser(self):
        parser = argparse.ArgumentParser(description="MLOps Framework CLI")
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # Pipeline commands
        pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline management")
        pipeline_subparsers = pipeline_parser.add_subparsers(dest="subcommand")
        
        # Pipeline list
        pipeline_list = pipeline_subparsers.add_parser("list", help="List pipelines")
        
        # Pipeline run
        pipeline_run = pipeline_subparsers.add_parser("run", help="Run a pipeline")
        pipeline_run.add_argument("pipeline", help="Pipeline name to run")
        pipeline_run.add_argument("--params", help="JSON string of parameters", default="{}")
        
        # Pipeline status
        pipeline_status = pipeline_subparsers.add_parser("status", help="Get pipeline status")
        pipeline_status.add_argument("run_id", help="Run ID to check")
        
        # Model commands
        model_parser = subparsers.add_parser("model", help="Model management")
        model_subparsers = model_parser.add_subparsers(dest="subcommand")
        
        # Model list
        model_list = model_subparsers.add_parser("list", help="List models")
        
        # Model register
        model_register = model_subparsers.add_parser("register", help="Register a model")
        model_register.add_argument("run_id", help="Run ID that produced the model")
        model_register.add_argument("name", help="Model name")
        model_register.add_argument("artifact_name", help="Artifact name containing the model")
        
        # Model promote
        model_promote = model_subparsers.add_parser("promote", help="Promote model to production")
        model_promote.add_argument("model_id", help="Model ID to promote")
        
        # Model compare
        model_compare = model_subparsers.add_parser("compare", help="Compare models")
        model_compare.add_argument("model_ids", help="Comma-separated list of model IDs to compare")
        
        # Deployment commands
        deploy_parser = subparsers.add_parser("deploy", help="Deployment management")
        deploy_subparsers = deploy_parser.add_subparsers(dest="subcommand")
        
        # Deploy model
        deploy_model = deploy_subparsers.add_parser("model", help="Deploy a model")
        deploy_model.add_argument("model_id", help="Model ID to deploy")
        deploy_model.add_argument("--env", help="Environment to deploy to", default="production")
        
        # List deployments
        deploy_list = deploy_subparsers.add_parser("list", help="List deployments")
        
        # Undeploy
        undeploy = deploy_subparsers.add_parser("undeploy", help="Remove a deployment")
        undeploy.add_argument("deployment_id", help="Deployment ID to remove")
        
        # Monitoring commands
        monitor_parser = subparsers.add_parser("monitor", help="Monitoring management")
        monitor_subparsers = monitor_parser.add_subparsers(dest="subcommand")
        
        # Monitor log metrics
        monitor_log = monitor_subparsers.add_parser("log", help="Log model metrics")
        monitor_log.add_argument("model_id", help="Model ID")
        monitor_log.add_argument("metrics", help="JSON string of metrics")
        monitor_log.add_argument("--drift", type=float, help="Data drift score", default=None)
        monitor_log.add_argument("--performance", type=float, help="Performance score", default=None)
        
        # Monitor alerts
        monitor_alerts = monitor_subparsers.add_parser("alerts", help="List active alerts")
        
        # Monitor history
        monitor_history = monitor_subparsers.add_parser("history", help="Get model performance history")
        monitor_history.add_argument("model_id", help="Model ID")
        
        return parser
        
    def run(self, args=None):
        args = self.parser.parse_args(args)
        
        if args.command is None:
            self.parser.print_help()
            return
            
        # Handle pipeline commands
        if args.command == "pipeline":
            self.handle_pipeline_command(args)
            
        # Handle model commands
        elif args.command == "model":
            self.handle_model_command(args)
            
        # Handle deployment commands
        elif args.command == "deploy":
            self.handle_deploy_command(args)
            
        # Handle monitoring commands
        elif args.command == "monitor":
            self.handle_monitor_command(args)
    
    def handle_pipeline_command(self, args):
        if args.subcommand is None:
            print("Pipeline subcommand required")
            return
            
        # List pipelines
        if args.subcommand == "list":
            pipelines = self._get_pipelines()
            if not pipelines:
                print("No pipelines found")
                return
                
            print("Available pipelines:")
            for name, versions in pipelines.items():
                print(f"  {name} (versions: {', '.join(versions)})")
                
        # Run pipeline
        elif args.subcommand == "run":
            try:
                pipeline = self._load_pipeline(args.pipeline)
                if pipeline is None:
                    return
                    
                params = json.loads(args.params)
                run_id = pipeline.run(params)
                print(f"Pipeline started with run ID: {run_id}")
                
            except Exception as e:
                print(f"Error running pipeline: {str(e)}")
                
        # Pipeline status
        elif args.subcommand == "status":
            self._check_run_status(args.run_id)
    
    def handle_model_command(self, args):
        if args.subcommand is None:
            print("Model subcommand required")
            return
            
        # Get first pipeline's DB for now - this could be improved
        db_path = self._get_first_pipeline_db()
        if not db_path:
            print("No pipelines found")
            return
            
        model_registry = ModelRegistry(db_path)
            
        # List models
        if args.subcommand == "list":
            models = self._list_models(model_registry)
            if not models:
                print("No models found")
                return
                
            print("Models:")
            for model in models:
                print(f"  {model['name']} (v{model['version']}) - ID: {model['id']} - Status: {model['status']}")
                
        # Register model
        elif args.subcommand == "register":
            try:
                # Load run artifacts
                pipeline = self._get_pipeline_by_run_id(args.run_id)
                if pipeline is None:
                    return
                    
                # Find the artifact
                artifact = self._get_artifact_from_run(pipeline, args.run_id, args.artifact_name)
                if artifact is None:
                    return
                    
                # Register the model
                model_id = model_registry.register_model(args.run_id, args.name, artifact)
                print(f"Model registered with ID: {model_id}")
                
            except Exception as e:
                print(f"Error registering model: {str(e)}")
                
        # Promote model
        elif args.subcommand == "promote":
            try:
                model_registry.promote_to_production(args.model_id)
                print(f"Model {args.model_id} promoted to production")
                
            except Exception as e:
                print(f"Error promoting model: {str(e)}")
                
        # Compare models
        elif args.subcommand == "compare":
            try:
                model_ids = args.model_ids.split(",")
                comparison = model_registry.compare_models(model_ids)
                
                if not comparison:
                    print("No models found for comparison")
                    return
                    
                print("Model Comparison:")
                for model in comparison:
                    print(f"\n{model['name']} (v{model['version']}) - ID: {model['id']}")
                    print("Metrics:")
                    for metric, value in model['metrics'].items():
                        print(f"  {metric}: {value}")
                
            except Exception as e:
                print(f"Error comparing models: {str(e)}")
    
    def handle_deploy_command(self, args):
        if args.subcommand is None:
            print("Deploy subcommand required")
            return
            
        # Get first pipeline's DB for now
        db_path = self._get_first_pipeline_db()
        if not db_path:
            print("No pipelines found")
            return
            
        model_registry = ModelRegistry(db_path)
        deployer = Deployment(model_registry)
            
        # Deploy model
        if args.subcommand == "model":
            try:
                deployment_id = deployer.deploy_model(args.model_id, args.env)
                print(f"Model deployed with ID: {deployment_id}")
                
            except Exception as e:
                print(f"Error deploying model: {str(e)}")
                
        # List deployments
        elif args.subcommand == "list":
            deployments = deployer.list_deployments()
            
            if not deployments:
                print("No active deployments found")
                return
                
            print("Active deployments:")
            for deployment in deployments:
                print(f"  {deployment['id']} - Model: {deployment['model_name']} (v{deployment['model_version']}) - Env: {deployment['environment']}")
                
        # Undeploy
        elif args.subcommand == "undeploy":
            try:
                success = deployer.undeploy(args.deployment_id)
                
                if success:
                    print(f"Deployment {args.deployment_id} successfully undeployed")
                else:
                    print(f"Deployment {args.deployment_id} not found")
                    
            except Exception as e:
                print(f"Error undeploying model: {str(e)}")
    
    def handle_monitor_command(self, args):
        if args.subcommand is None:
            print("Monitor subcommand required")
            return
            
        # Get first pipeline's DB for now
        db_path = self._get_first_pipeline_db()
        if not db_path:
            print("No pipelines found")
            return
            
        model_registry = ModelRegistry(db_path)
        monitor = ModelMonitor(model_registry)
            
        # Log metrics
        if args.subcommand == "log":
            try:
                metrics = json.loads(args.metrics)
                metric_id = monitor.log_metrics(
                    args.model_id, 
                    metrics, 
                    args.drift, 
                    args.performance
                )
                print(f"Metrics logged with ID: {metric_id}")
                
            except Exception as e:
                print(f"Error logging metrics: {str(e)}")
                
        # List alerts
        elif args.subcommand == "alerts":
            alerts = monitor.get_active_alerts()
            
            if not alerts:
                print("No active alerts found")
                return
                
            print("Active alerts:")
            for alert in alerts:
                print(f"  [{alert['alert_type']}] {alert['message']} (Model: {alert['model_id']})")
                
        # Model history
        elif args.subcommand == "history":
            try:
                history = monitor.get_model_performance_history(args.model_id)
                
                if not history:
                    print(f"No performance history found for model {args.model_id}")
                    return
                    
                print(f"Performance history for model {args.model_id}:")
                for entry in history:
                    print(f"\n  Timestamp: {entry['timestamp']}")
                    print("  Metrics:")
                    for metric, value in entry['metrics'].items():
                        print(f"    {metric}: {value}")
                    if entry['data_drift_score'] is not None:
                        print(f"  Data Drift Score: {entry['data_drift_score']}")
                    if entry['performance_score'] is not None:
                        print(f"  Performance Score: {entry['performance_score']}")
                
            except Exception as e:
                print(f"Error retrieving model history: {str(e)}")
    
    def _get_pipelines(self) -> Dict[str, List[str]]:
        """Get all available pipelines"""
        result = {}
        artifacts_dir = "artifacts"
        
        if not os.path.exists(artifacts_dir):
            return result
            
        for pipeline_name in os.listdir(artifacts_dir):
            pipeline_dir = os.path.join(artifacts_dir, pipeline_name)
            if os.path.isdir(pipeline_dir):
                versions = []
                for version in os.listdir(pipeline_dir):
                    if os.path.isdir(os.path.join(pipeline_dir, version)):
                        versions.append(version)
                if versions:
                    result[pipeline_name] = versions
        
        return result
        
    def _load_pipeline(self, pipeline_name: str) -> Pipeline:
        """Load a pipeline by name"""
        # This is a placeholder - in a real system, you'd have a way to load pipeline
        # definitions from a directory or database
        
        import importlib.util
        
        pipeline_file = f"pipelines/{pipeline_name}.py"
        
        if not os.path.exists(pipeline_file):
            print(f"Pipeline '{pipeline_name}' not found")
            return None
            
        try:
            spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Expect a get_pipeline function that returns the pipeline
            if hasattr(module, 'get_pipeline'):
                return module.get_pipeline()
            else:
                print(f"Pipeline file doesn't have a get_pipeline function")
                return None
                
        except Exception as e:
            print(f"Error loading pipeline: {str(e)}")
            return None
    
    def _check_run_status(self, run_id: str):
        """Check the status of a pipeline run"""
        # Find the pipeline for this run
        pipeline = self._get_pipeline_by_run_id(run_id)
        
        if pipeline is None:
            return
            
        # Query the database
        import sqlite3
        
        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT status, start_time, end_time FROM runs WHERE id = ?",
            (run_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            print(f"Run {run_id} not found")
            return
            
        status, start_time, end_time = result
        
        print(f"Run {run_id}:")
        print(f"  Status: {status}")
        print(f"  Started: {start_time}")
        if end_time:
            print(f"  Ended: {end_time}")
            
        # Get artifacts
        artifacts = self._get_run_artifacts(pipeline, run_id)
        
        if artifacts:
            print("\nArtifacts:")
            for artifact in artifacts:
                print(f"  {artifact['name']} (ID: {artifact['id']})")
    
    def _get_pipeline_by_run_id(self, run_id: str) -> Pipeline:
        """Find which pipeline a run belongs to"""
        # This is inefficient but works for a simple implementation
        pipelines = self._get_pipelines()
        
        for name, versions in pipelines.items():
            for version in versions:
                pipeline = Pipeline(name, version)
                
                # Check if this run exists in this pipeline
                import sqlite3
                
                try:
                    conn = sqlite3.connect(pipeline.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,))
                    
                    if cursor.fetchone():
                        conn.close()
                        return pipeline
                        
                    conn.close()
                except:
                    pass
        
        print(f"Run {run_id} not found in any pipeline")
        return None
    
    def _get_run_artifacts(self, pipeline: Pipeline, run_id: str) -> List[Dict]:
        """Get artifacts for a run"""
        import sqlite3
        
        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, name, path FROM artifacts WHERE run_id = ?",
            (run_id,)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"id": row[0], "name": row[1], "path": row[2]} for row in results]
    
    def _get_artifact_from_run(self, pipeline: Pipeline, run_id: str, artifact_name: str) -> Artifact:
        """Get a specific artifact from a run"""
        artifacts = self._get_run_artifacts(pipeline, run_id)
        
        for artifact in artifacts:
            if artifact["name"] == artifact_name:
                return Artifact.load(artifact["path"])
                
        print(f"Artifact '{artifact_name}' not found in run {run_id}")
        return None
    
    def _get_first_pipeline_db(self) -> str:
        """Get the first available pipeline DB path"""
        pipelines = self._get_pipelines()
        
        if not pipelines:
            return None
            
        name = list(pipelines.keys())[0]
        version = pipelines[name][0]
        
        pipeline = Pipeline(name, version)
        return pipeline.db_path
    
    def _list_models(self, model_registry: ModelRegistry) -> List[Dict]:
        """List all models in the registry"""
        import sqlite3
        
        conn = sqlite3.connect(model_registry.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, name, version, status FROM models"
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"id": row[0], "name": row[1], "version": row[2], "status": row[3]} for row in results]

if __name__ == "__main__":
    cli = CLI()
    cli.run()
