# api.py
from flask import Flask, request, jsonify
import json
import os
import sys
from typing import Dict, List, Any

from core.pipeline import Pipeline, PipelineStep, Artifact
from core.model_registry import ModelRegistry
from core.deployment import Deployment
from core.monitoring import ModelMonitor
from core.config import Config

app = Flask(__name__)
config = Config()

# Helper functions
def get_first_pipeline_db() -> str:
    """Get the first available pipeline DB path"""
    artifacts_dir = "artifacts"
        
    if not os.path.exists(artifacts_dir):
        return None
            
    for pipeline_name in os.listdir(artifacts_dir):
        pipeline_dir = os.path.join(artifacts_dir, pipeline_name)
        if os.path.isdir(pipeline_dir):
            for version in os.listdir(pipeline_dir):
                if os.path.isdir(os.path.join(pipeline_dir, version)):
                    pipeline = Pipeline(pipeline_name, version)
                    return pipeline.db_path
    
    return None

# Initialize services
db_path = get_first_pipeline_db()
model_registry = ModelRegistry(db_path) if db_path else None
deployer = Deployment(model_registry) if model_registry else None
monitor = ModelMonitor(model_registry) if model_registry else None

# API error handling
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500

# API routes - Pipelines
@app.route("/api/pipelines", methods=["GET"])
def list_pipelines():
    """List all available pipelines"""
    result = {}
    artifacts_dir = "artifacts"
    
    if not os.path.exists(artifacts_dir):
        return jsonify(result)
        
    for pipeline_name in os.listdir(artifacts_dir):
        pipeline_dir = os.path.join(artifacts_dir, pipeline_name)
        if os.path.isdir(pipeline_dir):
            versions = []
            for version in os.listdir(pipeline_dir):
                if os.path.isdir(os.path.join(pipeline_dir, version)):
                    versions.append(version)
            if versions:
                result[pipeline_name] = versions
    
    return jsonify(result)

@app.route("/api/pipelines/<name>", methods=["POST"])
def run_pipeline(name):
    """Run a pipeline"""
    params = request.json or {}
    
    import importlib.util
    
    pipeline_file = f"pipelines/{name}.py"
    
    if not os.path.exists(pipeline_file):
        return jsonify({"error": f"Pipeline '{name}' not found"}), 404
        
    try:
        spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Expect a get_pipeline function that returns the pipeline
        if hasattr(module, 'get_pipeline'):
            pipeline = module.get_pipeline()
            run_id = pipeline.run(params)
            return jsonify({"run_id": run_id})
        else:
            return jsonify({"error": "Pipeline file doesn't have a get_pipeline function"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Error running pipeline: {str(e)}"}), 500

@app.route("/api/runs/<run_id>", methods=["GET"])
def get_run_status(run_id):
    """Get the status of a pipeline run"""
    # Find the pipeline for this run
    artifacts_dir = "artifacts"
    
    if not os.path.exists(artifacts_dir):
        return jsonify({"error": "No pipelines found"}), 404
        
    for pipeline_name in os.listdir(artifacts_dir):
        pipeline_dir = os.path.join(artifacts_dir, pipeline_name)
        if os.path.isdir(pipeline_dir):
            for version in os.listdir(pipeline_dir):
                version_dir = os.path.join(pipeline_dir, version)
                if os.path.isdir(version_dir):
                    pipeline = Pipeline(pipeline_name, version)
                    
                    # Check if this run exists in this pipeline
                    import sqlite3
                    
                    try:
                        conn = sqlite3.connect(pipeline.db_path)
                        cursor = conn.cursor()
                        
                        cursor.execute(
                            "SELECT status, start_time, end_time, parameters FROM runs WHERE id = ?",
                            (run_id,)
                        )
                        
                        result = cursor.fetchone()
                        
                        if result:
                            status, start_time, end_time, params_json = result
                            
                            # Get artifacts
                            cursor.execute(
                                "SELECT id, name, path, metadata FROM artifacts WHERE run_id = ?",
                                (run_id,)
                            )
                            
                            artifacts = [
                                {
                                    "id": row[0], 
                                    "name": row[1], 
                                    "path": row[2],
                                    "metadata": json.loads(row[3]) if row[3] else {}
                                } 
                                for row in cursor.fetchall()
                            ]
                            
                            conn.close()
                            
                            return jsonify({
                                "run_id": run_id,
                                "pipeline": pipeline_name,
                                "version": version,
                                "status": status,
                                "start_time": start_time,
                                "end_time": end_time,
                                "parameters": json.loads(params_json) if params_json else {},
                                "artifacts": artifacts
                            })
                        
                        conn.close()
                    except Exception as e:
                        pass
    
    return jsonify({"error": f"Run {run_id} not found"}), 404

# API routes - Models
@app.route("/api/models", methods=["GET"])
def list_models():
    """List all models in the registry"""
    if not model_registry:
        return jsonify({"error": "Model registry not initialized"}), 500
        
    import sqlite3
    
    conn = sqlite3.connect(model_registry.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, name, version, status, metrics, created_at FROM models"
    )
    
    results = cursor.fetchall()
    conn.close()
    
    models = [
        {
            "id": row[0], 
            "name": row[1], 
            "version": row[2], 
            "status": row[3],
            "metrics": json.loads(row[4]) if row[4] else {},
            "created_at": row[5]
        } 
        for row in results
    ]
    
    return jsonify(models)

@app.route("/api/models", methods=["POST"])
def register_model():
    """Register a model"""
    if not model_registry:
        return jsonify({"error": "Model registry not initialized"}), 500
        
    data = request.json
    
    if not data or "run_id" not in data or "name" not in data or "artifact_name" not in data:
        return jsonify({"error": "Missing required fields: run_id, name, artifact_name"}), 400
        
    run_id = data["run_id"]
    name = data["name"]
    artifact_name = data["artifact_name"]
    
    # Find the pipeline for this run
    artifacts_dir = "artifacts"
    
    pipeline = None
    for pipeline_name in os.listdir(artifacts_dir):
        pipeline_dir = os.path.join(artifacts_dir, pipeline_name)
        if os.path.isdir(pipeline_dir):
            for version in os.listdir(pipeline_dir):
                version_dir = os.path.join(pipeline_dir, version)
                if os.path.isdir(version_dir):
                    p = Pipeline(pipeline_name, version)
                    
                    # Check if this run exists in this pipeline
                    import sqlite3
                    
                    try:
                        conn = sqlite3.connect(p.db_path)
                        cursor = conn.cursor()
                        
                        cursor.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,))
                        
                        if cursor.fetchone():
                            pipeline = p
                            conn.close()
                            break
                            
                        conn.close()
                    except:
                        pass
                        
            if pipeline:
                break
    
    if not pipeline:
        return jsonify({"error": f"Run {run_id} not found"}), 404
        
    # Find the artifact
    conn = sqlite3.connect(pipeline.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, path FROM artifacts WHERE run_id = ? AND name = ?",
        (run_id, artifact_name)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({"error": f"Artifact '{artifact_name}' not found in run {run_id}"}), 404
        
    artifact_id, artifact_path = result
    
    try:
        artifact = Artifact.load(artifact_path)
        metrics = data.get("metrics", {})
        
        model_id = model_registry.register_model(run_id, name, artifact, metrics)
        
        return jsonify({"model_id": model_id})
        
    except Exception as e:
        return jsonify({"error": f"Error registering model: {str(e)}"}), 500

@app.route("/api/models/<model_id>/promote", methods=["POST"])
def promote_model(model_id):
    """Promote a model to production"""
    if not model_registry:
        return jsonify({"error": "Model registry not initialized"}), 500
        
    try:
        model_registry.promote_to_production(model_id)
        return jsonify({"success": True, "model_id": model_id})
        
    except Exception as e:
        return jsonify({"error": f"Error promoting model: {str(e)}"}), 500

@app.route("/api/models/compare", methods=["POST"])
def compare_models():
    """Compare models"""
    if not model_registry:
        return jsonify({"error": "Model registry not initialized"}), 500
        
    data = request.json
    
    if not data or "model_ids" not in data:
        return jsonify({"error": "Missing required field: model_ids"}), 400
        
    model_ids = data["model_ids"]
    
    try:
        comparison = model_registry.compare_models(model_ids)
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({"error": f"Error comparing models: {str(e)}"}), 500

# API routes - Deployments
@app.route("/api/deployments", methods=["GET"])
def list_deployments():
    """List all deployments"""
    if not deployer:
        return jsonify({"error": "Deployment service not initialized"}), 500
        
    environment = request.args.get("environment")
    
    deployments = deployer.list_deployments(environment)
    return jsonify(deployments)

@app.route("/api/deployments", methods=["POST"])
def deploy_model():
    """Deploy a model"""
    if not deployer:
        return jsonify({"error": "Deployment service not initialized"}), 500
        
    data = request.json
    
    if not data or "model_id" not in data:
        return jsonify({"error": "Missing required field: model_id"}), 400
        
    model_id = data["model_id"]
    environment = data.get("environment", "production")
    
    try:
        deployment_id = deployer.deploy_model(model_id, environment)
        return jsonify({"deployment_id": deployment_id})
        
    except Exception as e:
        return jsonify({"error": f"Error deploying model: {str(e)}"}), 