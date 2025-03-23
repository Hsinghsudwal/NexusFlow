# model_registry/registry.py
import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

class ModelRegistry:
    def __init__(self, registry_path: str, db_manager=None):
        """Initialize the model registry with a path to store models and artifacts."""
        self.registry_path = registry_path
        self.models_path = os.path.join(registry_path, "models")
        self.artifacts_path = os.path.join(registry_path, "artifacts")
        self.db_manager = db_manager
        
        # Create directory structure if it doesn't exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)
        
    def register_model(self, 
                      model_path: str, 
                      model_name: str,
                      run_id: str,
                      framework: str,
                      performance_metrics: Dict[str, float],
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the trained model file
            model_name: Name of the model
            run_id: ID of the run that created this model
            framework: ML framework used (e.g., sklearn, tensorflow)
            performance_metrics: Dictionary of performance metrics
            metadata: Additional metadata for the model
            
        Returns:
            model_version: Version string of the registered model
        """
        # Generate a version based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"{timestamp}_{str(uuid.uuid4())[:8]}"
        
        # Create model directory
        model_dir = os.path.join(self.models_path, model_name, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file to registry
        model_filename = os.path.basename(model_path)
        dest_path = os.path.join(model_dir, model_filename)
        shutil.copy2(model_path, dest_path)
        
        # Create metadata file
        metadata_dict = metadata or {}
        metadata_dict.update({
            "name": model_name,
            "version": model_version,
            "run_id": run_id,
            "framework": framework,
            "performance_metrics": performance_metrics,
            "created_at": datetime.now().isoformat(),
            "status": "registered",  # Initial status
            "model_file": model_filename
        })
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # If DB manager exists, register in the database too
        if self.db_manager:
            # Log the model file as an artifact first
            artifact = self.db_manager.log_artifact(
                run_id=run_id,
                name=f"{model_name}_model",
                artifact_type="model",
                path=dest_path,
                metadata={"framework": framework, "filename": model_filename}
            )
            
            if artifact:
                # Register the model in the database
                self.db_manager.register_model(
                    name=model_name,
                    version=model_version,
                    artifact_id=artifact.id,
                    performance=performance_metrics
                )
        
        return model_version
    
    def promote_to_production(self, model_name: str, model_version: str) -> bool:
        """
        Promote a model to production status.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model to promote
            
        Returns:
            bool: True if promotion was successful
        """
        model_dir = os.path.join(self.models_path, model_name, model_version)
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return False
        
        # Read metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update status
        metadata["status"] = "production"
        metadata["promoted_at"] = datetime.now().isoformat()
        
        # Write updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create a symlink for easy access to the current production model
        prod_link = os.path.join(self.models_path, model_name, "production")
        
        # Remove existing symlink if it exists
        if os.path.islink(prod_link):
            os.unlink(prod_link)
        
        # Create relative symlink
        os.symlink(os.path.relpath(model_dir, os.path.dirname(prod_link)), prod_link)
        
        # Update database if available
        if self.db_manager:
            model = self.get_model_by_version(model_name, model_version)
            if model and "db_id" in model:
                self.db_manager.promote_model_to_production(model["db_id"])
        
        return True
    
    def archive_model(self, model_name: str, model_version: str) -> bool:
        """
        Archive a model version.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model to archive
            
        Returns:
            bool: True if archiving was successful
        """
        model_dir = os.path.join(self.models_path, model_name, model_version)
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return False
        
        # Read metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update status
        metadata["status"] = "archived"
        metadata["archived_at"] = datetime.now().isoformat()
        
        # Write updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def get_model_by_version(self, model_name: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by version."""
        metadata_path = os.path.join(self.models_path, model_name, model_version, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the current production model metadata."""
        prod_link = os.path.join(self.models_path, model_name, "production")
        
        if not os.path.islink(prod_link):
            # Try database as fallback
            if self.db_manager:
                model = self.db_manager.get_latest_production_model(model_name)
                if model:
                    return {
                        "name": model.name,
                        "version": model.version,
                        "status": model.status,
                        "performance_metrics": model.performance,
                        "created_at": model.created_at.isoformat(),
                        "db_id": model.id
                    }
            return None
        
        metadata_path = os.path.join(prod_link, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_model_file_path(self, model_name: str, model_version: str = None) -> Optional[str]:
        """
        Get the file path for a model.
        If model_version is None, returns the production model.
        """
        if model_version is None:
            # Get production model
            prod_metadata = self.get_production_model(model_name)
            if not prod_metadata:
                return None
            model_version = prod_metadata["version"]
        
        model_metadata = self.get_model_by_version(model_name, model_version)
        if not model_metadata:
            return None
        
        return os.path.join(self.models_path, model_name, model_version, model_metadata["model_file"])
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model with their metadata."""
        model_base_dir = os.path.join(self.models_path, model_name)
        
        if not os.path.exists(model_base_dir):
            return []
        
        versions = []
        for version_dir in os.listdir(model_base_dir):
            # Skip the production symlink
            if version_dir == "production":
                continue
            
            metadata_path = os.path.join(model_base_dir, version_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    versions.append(metadata)
        
        # Sort by created_at timestamp (newest first)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return versions
    
    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions by their performance metrics."""
        model1 = self.get_model_by_version(model_name, version1)
        model2 = self.get_model_by_version(model_name, version2)
        
        if not model1 or not model2:
            return {"error": "One or both models not found"}
        
        metrics1 = model1.get("performance_metrics", {})
        metrics2 = model2.get("performance_metrics", {})
        
        # Get all unique metric names
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        # Calculate differences
        comparison = {
            "model1": {
                "version": version1,
                "metrics": metrics1
            },
            "model2": {
                "version": version2,
                "metrics": metrics2
            },
            "differences": {}
        }
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, None)
            val2 = metrics2.get(metric, None)
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct_change = (diff / val1) * 100 if val1 != 0 else float('inf')
                comparison["differences"][metric] = {
                    "absolute": diff,
                    "percentage": pct_change
                }
            else:
                comparison["differences"][metric] = {
                    "absolute": None,
                    "percentage": None,
                    "note": "Metric not present in both models"
                }
        
        return comparison

# model_registry/artifact_store.py
import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

class ArtifactStore:
    def __init__(self, store_path: str, db_manager