# deployment/deployer.py
import os
import json
import datetime
import subprocess
import shutil
from typing import Dict, Any, Optional, List, Union

from ..core.model_registry import ModelRegistry


class ModelDeployer:
    """
    Model deployment manager that handles deploying models to different environments.
    
    Supported deployment targets:
    - local: Deploys the model for local serving
    - docker: Packages the model into a Docker container
    - kubernetes: Deploys the model to a Kubernetes cluster
    """
    
    def __init__(self, deployment_dir: str = "deployments"):
        self.deployment_dir = deployment_dir
        os.makedirs(deployment_dir, exist_ok=True)
        self.model_registry = ModelRegistry()
        self.deployment_registry_file = os.path.join(deployment_dir, "deployment_registry.json")
        
        # Initialize deployment registry if it doesn't exist
        if not os.path.exists(self.deployment_registry_file):
            with open(self.deployment_registry_file, 'w') as f:
                json.dump({"deployments": {}}, f, indent=2)
                
    def deploy_model(self, 
                   model_name: str, 
                   model_version: Optional[str] = None,
                   deployment_target: str = "local",
                   deployment_name: Optional[str] = None,
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deploy a model to the specified target.
        
        Args:
            model_name: Name of the model to deploy
            model_version: Model version, if None uses the latest
            deployment_target: Where to deploy ('local', 'docker', 'kubernetes')
            deployment_name: Custom name for the deployment
            config: Deployment-specific configuration
            
        Returns:
            Deployment information
        """
        # Get the model from the registry
        model_info = self.model_registry.get_model(model_name, model_version)
        model_version = model_info["version"]  # In case None was passed in
        
        # Create deployment name if not provided
        if deployment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            deployment_name = f"{model_name}_{model_version}_{timestamp}"
            
        # Create deployment directory
        deployment_dir = os.path.join(self.deployment_dir, deployment_name)
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Merge with default configuration
        default_config = {
            "port": 8000,
            "monitoring_enabled": True,
            "log_level": "info"
        }
        deployment_config = {**default_config, **(config or {})}
        
        # Execute deployment based on target
        if deployment_target == "local":
            deployment_info = self._deploy_local(
                model_info, deployment_name, deployment_dir, deployment_config
            )
        elif deployment_target == "docker":
            deployment_info = self._deploy_docker(
                model_info, deployment_name, deployment_dir, deployment_config
            )
        elif deployment_target == "kubernetes":
            deployment_info = self._deploy_kubernetes(
                model_info, deployment_name, deployment_dir, deployment_config
            )
        else:
            raise ValueError(f"Unsupported deployment target: {deployment_target}")
            
        # Register the deployment
        self._register_deployment(deployment_name, model_name, model_version, 
                                deployment_target, deployment_config, deployment_info)
        
        return deployment_info
    
    def _deploy_local(self, 
                     model_info: Dict[str, Any],
                     deployment_name: str,
                     deployment_dir: str,
                     config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model for local serving."""
        # Copy model to deployment directory
        model_path = model_info["absolute_path"]
        local_model_path = os.path.join(deployment_dir, "model")
        
        if os.path.isdir(model_path):
            # Copy directory
            shutil.copytree(model_path, local_model_path, dirs_exist_ok=True)
        else:
            # Copy file
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            shutil.copy2(model_path, local_model_path)
            
        # Create serving script
        serving_script = self._generate_local_serving_script(
            deployment_name, local_model_path, config
        )
        
        serving_script_path = os.path.join(deployment_dir, "serve.py")
        with open(serving_script_path, 'w') as f:
            f.write(serving_script)
            
        # Create start script
        start_script = f"""#!/bin/bash
cd {os.path.abspath(deployment_dir)}
python serve.py
"""
        
        start_script_path = os.path.join(deployment_dir, "start.sh")
        with open(start_script_path, 'w') as f:
            f.write(start_script)
            
        # Make the script executable
        os.chmod(start_script_path, 0o755)
        
        # Return deployment info
        return {
            "deployment_type": "local",
            "model_name": model_info["name"],
            "model_version": model_info["version"],
            "deployment_directory": deployment_dir,
            "start_script": start_script_path,
            "port": config["port"],
            "endpoint": f"http://localhost:{config['port']}/predict",
            "status": "deployed"
        }
    
    def _deploy_docker(self, 
                      model_info: Dict[str, Any],
                      deployment_name: str,
                      deployment_dir: str,
                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model as a Docker container."""
        # Copy model to deployment directory
        model_path = model_info["absolute_path"]
        local_model_path = os.path.join(deployment_dir, "model")
        
        if os.path.isdir(model_path):
            # Copy directory
            shutil.copytree(model_path, local_model_path, dirs_exist_ok=True)
        else:
            # Copy file
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            shutil.copy2(model_path, local_model_path)
            
        # Create serving script
        serving_script = self._generate_local_serving_script(
            deployment_name, "/app/model", config
        )
        
        serving_script_path = os.path.join(deployment_dir, "serve.py")
        with open(serving_script_path, 'w') as f:
            f.write(serving_script)
            
        # Create Dockerfile
        dockerfile = f"""FROM python:3.9-slim

WORKDIR /app

# Copy model and serving script
COPY model /app/model
COPY serve.py /app/serve.py

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pandas numpy joblib

# Expose the port
EXPOSE {config['port']}

# Start the service
CMD ["python", "serve.py"]
"""
        
        dockerfile_path = os.path.join(deployment_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile)
            
        # Create docker build and run scripts
        docker_image_name = f"mlops-model-{deployment_name.lower()}"
        
        docker_build_script = f"""#!/bin/bash
cd {os.path.abspath(deployment_dir)}
docker build -t {docker_image_name} .
"""
        
        docker_run_script = f"""#!/bin/bash
docker run -d --name {deployment_name} -p {config['port']}:{config['port']} {docker_image_name}
"""
        
        docker_build_path = os.path.join(deployment_dir, "build_docker.sh")
        docker_run_path = os.path.join(deployment_dir, "run_docker.sh")
        
        with open(docker_build_path, 'w') as f:
            f.write(docker_build_script)
            
        with open(docker_run_path, 'w') as f:
            f.write(docker_run_script)
            
        # Make scripts executable
        os.chmod(docker_build_path, 0o755)
        os.chmod(docker_run_path, 0o755)
        
        # Return deployment info
        return {
            "deployment_type": "docker",
            "model_name": model_info["name"],
            "model_version": model_info["version"],
            "deployment_directory": deployment_dir,
            "docker_image": docker_image_name,
            "build_script": docker_build_path,
            "run_script": docker_run_path,
            "port": config["port"],
            "endpoint": f"http://localhost:{config['port']}/predict",
            "status": "ready_to_build"
        }
    
    def _deploy_kubernetes(self, 
                         model_info: Dict[str, Any],
                         deployment_name: str,
                         deployment_dir: str,
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to Kubernetes."""
        # First create a Docker deployment
        docker_info = self._deploy_docker(
            model_info, deployment_name, deployment_dir, config
        )
        
        # Create Kubernetes deployment YAML
        k8s_deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  labels:
    app: {deployment_name}
spec:
  replicas: {config.get('replicas', 1)}
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
      labels:
        app: {deployment_name}
    spec:
      containers:
      - name: model-server
        image: {docker_info['docker_image']}
        ports:
        - containerPort: {config['port']}
        resources:
          requests:
            memory: "{config.get('memory_request', '256Mi')}"
            cpu: "{config.get('cpu_request', '100m')}"
          limits:
            memory: "{config.get('memory_limit', '512Mi')}"
            cpu: "{config.get('cpu_limit', '500m')}"
---
apiVersion: v1
kind: Service
metadata:
  name: {deployment_name}-service
spec:
  selector:
    app: {deployment_name}
  ports:
  - port: 80
    targetPort: {config['port']}
  type: ClusterIP
"""
        
        if config.get('create_ingress', False):
            k8s_deployment += f"""
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {deployment_name}-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: {config.get('ingress_host', deployment_name + '.example.com')}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {deployment_name}-service
            port:
              number: 80
"""
        
        k8s_file_path = os.path.join(deployment_dir, "kubernetes.yaml")
        with open(k8s_file_path, 'w') as f:
            f.write(k8s_deployment)
            
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# First build the Docker image
{os.path.abspath(docker_info['build_script'])}

# Push to registry if specified
# kubectl create secret docker-registry regcred --docker-server=<your-registry-server> --docker-username=<your-name> --docker-password=<your-pword> --docker-email=<your-email>

# Deploy to Kubernetes
kubectl apply -f {os.path.abspath(k8s_file_path)}
"""
        
        deploy_script_path = os.path.join(deployment_dir, "deploy_k8s.sh")
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
            
        # Make script executable
        os.chmod(deploy_script_path, 0o755)
        
        # Return deployment info
        return {
            "deployment_type": "kubernetes",
            "model_name": model_info["name"],
            "model_version": model_info["version"],
            "deployment_directory": deployment_dir,
            "docker_image": docker_info["docker_image"],
            "kubernetes_file": k8s_file_path,
            "deploy_script": deploy_script_path,
            "service_name": f"{deployment_name}-service",
            "status": "ready_to_deploy"
        }
    
    def _generate_local_serving_script(self, 
                                     deployment_name: str,
                                     model_path: str,
                                     config: Dict[str, Any]) -> str:
        """Generate a Python script for model serving."""
        script = f"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Union, Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, "{config.get('log_level', 'INFO').upper()}"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger("{deployment_name}")

# Initialize FastAPI app
app = FastAPI(
    title="{deployment_name} Model API",
    description="API for serving machine learning model predictions",
    version="1.0"
)

# Load the model
logger.info(f"Loading model from {model_path}")
try:
    model = joblib.load(os.path.join("{model_path}", "model.pkl"))
    logger.info(f"Model loaded successfully: {{type(model).__name__}}")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    # Try loading as a directory of artifacts
    try:
        model = joblib.load("{model_path}")
        logger.info(f"Model loaded successfully as full path: {{type(model).__name__}}")
    except Exception as e2:
        logger.error(f"Failed to load model from full path: {{e2}}")
        sys.exit(1)

# Define request/response models
class PredictionRequest(BaseModel):
    data: Union[List[List[float]], List[Dict[str, float]]]
    feature_names: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    predictions: List
    prediction_probabilities: Optional[List[List[float]]] = None
    model_info: Dict[str, str]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Log request (excluding data for privacy)
        logger.info(f"Received prediction request with {{len(request.data)}} samples")
        
        # Convert input data to DataFrame if feature names are provided
        if request.feature_names:
            if isinstance(request.data[0], dict):
                # Data already in dict format
                df = pd.DataFrame(request.data)
            else:
                # Data in list format, convert to DataFrame
                df = pd.DataFrame(request.data, columns=request.feature_names)
        else:
            # If no feature names, assume data is properly formatted
            if isinstance(request.data[0], dict):
                df = pd.DataFrame(request.data)
            else:
                df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        # Get prediction probabilities if the model supports it
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df).tolist()
            except:
                logger.warning("Model has predict_proba method but it failed")
        
        # Create response
        response = PredictionResponse(
            predictions=predictions,
            prediction_probabilities=probabilities,
            model_info={{
                "model_type": type(model).__name__,
                "deployment_name": "{deployment_name}"
            }}
        )
        
        # Log successful prediction
        logger.info(f"Successfully generated predictions")
        
        return response
    
    except Exception as e:
        logger.error(f"Error during prediction: {{str(e)}}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {{str(e)}}")

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "model": "{deployment_name}"}}

@app.get("/")
async def root():
    return {{
        "model_name": "{deployment_name}",
        "description": "Model prediction API",
        "endpoints": [
            {{"path": "/predict", "method": "POST", "description": "Get model predictions"}},
            {{"path": "/health", "method": "GET", "description": "Health check"}}
        ]
    }}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port={config['port']})
"""
        return script
    
    def _register_deployment(self, 
                           deployment_name: str,
                           model_name: str,
                           model_version: str,
                           deployment_target: str,
                           config: Dict[str, Any],
                           deployment_info: Dict[str, Any]) -> None:
        """Register a deployment in the deployment registry."""
        with open(self.deployment_registry_file, 'r') as f:
            registry = json.load(f)
            
        # Add deployment
        registry["deployments"][deployment_name] = {
            "model_name": model_name,
            "model_version": model_version,
            "deployment_target": deployment_target,
            "config": config,
            "deployment_info": deployment_info,
            "created_at": datetime.datetime.now().isoformat(),
            "status": deployment_info.get("status", "unknown")
        }
        
        with open(self.deployment_registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def list_deployments(self) -> Dict[str, Any]:
        """List all deployments."""
        with open(self.deployment_registry_file, 'r') as f:
            registry = json.load(f)
            
        return registry["deployments"]
    
    def get_deployment(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment details."""
        with open(self.deployment_registry_file, 'r') as f:
            registry = json.load(f)
            
        if deployment_name not in registry["deployments"]:
            raise ValueError(f"Deployment not found: {deployment_name}")
            
        return registry["deployments"][deployment_name]
    
    def update_deployment_status(self, deployment_name: str, status: str) -> None:
        """Update the status of a deployment."""
        with open(self.deployment_registry_file, 'r') as f:
            registry = json.load(f)
            
        if deployment_name not in registry["deployments"]:
            raise ValueError(f"Deployment not found: {deployment_name}")
            
        registry["deployments"][deployment_name]["status"] = status
        registry["deployments"][deployment_name]["updated_at"] = datetime.datetime.now().isoformat()
        
        with open(self.deployment_registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def undeploy(self, deployment_name: str) -> Dict[str, Any]:
        """Remove a deployment."""
        deployment = self.get_deployment(deployment_name)
        deployment_type = deployment["deployment_target"]
        
        # Execute undeployment scripts based on type
        if deployment_type == "docker":
            undeploy_script = f"""#!/bin/bash
docker stop {deployment_name} || true
docker rm {deployment_name} || true
docker rmi {deployment['deployment_info']['docker_image']} || true
"""
            undeploy_path = os.path.join(deployment["deployment_info"]["deployment_directory"], "undeploy.sh")
            with open(undeploy_path, 'w') as f:
                f.write(undeploy_script)
                
            os.chmod(undeploy_path, 0o755)
            
            try:
                subprocess.run([undeploy_path], check=True)
            except subprocess.CalledProcessError:
                print(f"Warning: Error while undeploying Docker container {deployment_name}")
                
        elif deployment_type == "kubernetes":
            undeploy_script = f"""#!/bin/bash
kubectl delete -f {deployment['deployment_info']['kubernetes_file']} || true
"""
            undeploy_path = os.path.join(deployment["deployment_info"]["deployment_directory"], "undeploy.sh")
            with open(undeploy_path, 'w') as f:
                f.write(undeploy_script)
                
            os.chmod(undeploy_path, 0o755)
            
            try:
                subprocess.run([undeploy_path], check=True)
            except subprocess.Calle