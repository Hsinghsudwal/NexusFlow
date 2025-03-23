# core/pipeline.py
import os
import json
import uuid
import datetime
import sqlite3
import pickle
from typing import Dict, List, Any, Callable, Optional
import yaml
import hashlib

class Artifact:
    def __init__(self, name: str, data: Any, metadata: Dict = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now().isoformat()
        
    def save(self, artifacts_dir: str) -> str:
        """Save artifact to disk and return the path"""
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save the data
        artifact_path = os.path.join(artifacts_dir, f"{self.name}_{self.id}")
        with open(artifact_path, 'wb') as f:
            pickle.dump(self.data, f)
            
        # Save metadata
        metadata = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "path": artifact_path
        }
        
        with open(f"{artifact_path}_meta.json", 'w') as f:
            json.dump(metadata, f)
            
        return artifact_path
    
    @classmethod
    def load(cls, artifact_path: str):
        """Load an artifact from disk"""
        # Load metadata
        with open(f"{artifact_path}_meta.json", 'r') as f:
            metadata = json.load(f)
            
        # Load data
        with open(artifact_path, 'rb') as f:
            data = pickle.load(f)
            
        artifact = cls(metadata["name"], data, metadata["metadata"])
        artifact.id = metadata["id"]
        artifact.created_at = metadata["created_at"]
        
        return artifact

class PipelineStep:
    def __init__(self, name: str, function: Callable, inputs: List[str] = None, outputs: List[str] = None):
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step function with inputs from context"""
        # Extract inputs from context
        input_data = {input_name: context[input_name] for input_name in self.inputs if input_name in context}
        
        # Execute the function
        results = self.function(**input_data)
        
        # Handle single output vs dictionary of outputs
        if not isinstance(results, dict):
            if len(self.outputs) == 1:
                results = {self.outputs[0]: results}
            else:
                raise ValueError(f"Step {self.name} returned a single value but expected {len(self.outputs)} outputs")
                
        # Update context with results
        for output_name in self.outputs:
            if output_name in results:
                context[output_name] = results[output_name]
                
        return results

class Pipeline:
    def __init__(self, name: str, version: str = "0.1.0"):
        self.name = name
        self.version = version
        self.steps = []
        self.artifacts_dir = os.path.join("artifacts", self.name, self.version)
        self.db_path = os.path.join("db", f"{self.name}.db")
        self.init_db()
        
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline"""
        self.steps.append(step)
        return self
        
    def init_db(self):
        """Initialize the SQLite database for tracking runs and artifacts"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create runs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            pipeline_name TEXT,
            pipeline_version TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            parameters TEXT
        )
        ''')
        
        # Create artifacts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            name TEXT,
            path TEXT,
            created_at TEXT,
            metadata TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
        ''')
        
        # Create models table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            name TEXT,
            version TEXT,
            artifact_id TEXT,
            metrics TEXT,
            status TEXT,
            created_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id),
            FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def run(self, parameters: Dict[str, Any] = None) -> str:
        """Execute the pipeline and track results"""
        run_id = str(uuid.uuid4())
        os.makedirs(os.path.join(self.artifacts_dir, run_id), exist_ok=True)
        
        parameters = parameters or {}
        start_time = datetime.datetime.now().isoformat()
        
        # Record run in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, self.name, self.version, "RUNNING", start_time, None, json.dumps(parameters))
        )
        conn.commit()
        
        try:
            # Initialize context with parameters
            context = parameters.copy()
            
            # Execute each step
            for step in self.steps:
                print(f"Executing step: {step.name}")
                results = step.execute(context)
                
                # Save any artifacts that were created
                for output_name in step.outputs:
                    if output_name in context and isinstance(context[output_name], Artifact):
                        artifact = context[output_name]
                        artifact_path = artifact.save(os.path.join(self.artifacts_dir, run_id))
                        
                        # Record artifact in database
                        cursor.execute(
                            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
                            (artifact.id, run_id, artifact.name, artifact_path, 
                             artifact.created_at, json.dumps(artifact.metadata))
                        )
            
            # Update run status to completed
            end_time = datetime.datetime.now().isoformat()
            cursor.execute(
                "UPDATE runs SET status = ?, end_time = ? WHERE id = ?",
                ("COMPLETED", end_time, run_id)
            )
            conn.commit()
            
            print(f"Pipeline completed successfully: {run_id}")
            return run_id
            
        except Exception as e:
            # Update run status to failed
            cursor.execute(
                "UPDATE runs SET status = ?, end_time = ? WHERE id = ?",
                ("FAILED", datetime.datetime.now().isoformat(), run_id)
            )
            conn.commit()
            print(f"Pipeline failed: {str(e)}")
            raise
        finally:
            conn.close()

# model_registry.py
class ModelRegistry:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def register_model(self, run_id: str, name: str, artifact: Artifact, metrics: Dict = None) -> str:
        """Register a model in the registry"""
        model_id = str(uuid.uuid4())
        version = self._get_next_version(name)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model_id, run_id, name, version, artifact.id, 
             json.dumps(metrics or {}), "STAGING", datetime.datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        return model_id
    
    def promote_to_production(self, model_id: str) -> bool:
        """Promote a model to production status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, demote any current production models for this model type
        cursor.execute("SELECT name FROM models WHERE id = ?", (model_id,))
        model_name = cursor.fetchone()[0]
        
        cursor.execute(
            "UPDATE models SET status = ? WHERE name = ? AND status = ?",
            ("ARCHIVED", model_name, "PRODUCTION")
        )
        
        # Then promote the new model
        cursor.execute(
            "UPDATE models SET status = ? WHERE id = ?",
            ("PRODUCTION", model_id)
        )
        
        conn.commit()
        conn.close()
        return True
    
    def get_production_model(self, name: str) -> Dict:
        """Get the current production model for a given name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, version, artifact_id, metrics, created_at FROM models WHERE name = ? AND status = ?",
            (name, "PRODUCTION")
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        return {
            "id": result[0],
            "name": name,
            "version": result[1],
            "artifact_id": result[2],
            "metrics": json.loads(result[3]),
            "created_at": result[4]
        }
        
    def get_model_artifact(self, model_id: str) -> Artifact:
        """Get the artifact for a model by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.path 
            FROM artifacts a
            JOIN models m ON a.id = m.artifact_id
            WHERE m.id = ?
        """, (model_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        return Artifact.load(result[0])
    
    def _get_next_version(self, model_name: str) -> str:
        """Get the next version number for a model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT MAX(CAST(version AS INTEGER)) FROM models WHERE name = ?",
            (model_name,)
        )
        
        result = cursor.fetchone()[0]
        conn.close()
        
        if result is None:
            return "1"
        else:
            return str(int(result) + 1)
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """Compare metrics between multiple models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        models = []
        for model_id in model_ids:
            cursor.execute(
                "SELECT name, version, metrics FROM models WHERE id = ?",
                (model_id,)
            )
            result = cursor.fetchone()
            if result:
                models.append({
                    "id": model_id,
                    "name": result[0],
                    "version": result[1],
                    "metrics": json.loads(result[2])
                })
        
        conn.close()
        return models

# deployment.py
class Deployment:
    def __init__(self, model_registry: ModelRegistry, deployment_config: Dict = None):
        self.model_registry = model_registry
        self.config = deployment_config or {}
        self.deployments = {}
        
    def deploy_model(self, model_id: str, environment: str = "production") -> bool:
        """Deploy a model to the specified environment"""
        # Get model information
        conn = sqlite3.connect(self.model_registry.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name, version, artifact_id FROM models WHERE id = ?",
            (model_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Model with ID {model_id} not found")
            
        model_name, model_version, artifact_id = result
        
        # Get the model artifact
        model_artifact = self.model_registry.get_model_artifact(model_id)
        
        # Record deployment
        deployment_id = str(uuid.uuid4())
        self.deployments[deployment_id] = {
            "id": deployment_id,
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "environment": environment,
            "status": "ACTIVE",
            "deployed_at": datetime.datetime.now().isoformat()
        }
        
        print(f"Model {model_name} (version {model_version}) deployed to {environment}")
        return deployment_id
        
    def list_deployments(self, environment: str = None) -> List[Dict]:
        """List all active deployments, optionally filtered by environment"""
        deployments = list(self.deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d["environment"] == environment]
            
        return deployments
        
    def undeploy(self, deployment_id: str) -> bool:
        """Remove a deployment"""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["status"] = "INACTIVE"
            print(f"Deployment {deployment_id} has been deactivated")
            return True
        return False

# monitoring.py
class ModelMonitor:
    def __init__(self, model_registry: ModelRegistry, db_path: str = None):
        self.model_registry = model_registry
        self.db_path = db_path or model_registry.db_path
        self.init_monitoring_db()
        
    def init_monitoring_db(self):
        """Initialize monitoring database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            timestamp TEXT,
            metrics TEXT,
            data_drift_score REAL,
            performance_score REAL,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            alert_type TEXT,
            message TEXT,
            timestamp TEXT,
            resolved INTEGER,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_metrics(self, model_id: str, metrics: Dict, data_drift_score: float = None,
                   performance_score: float = None) -> str:
        """Log model performance metrics"""
        metric_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?, ?)",
            (metric_id, model_id, timestamp, json.dumps(metrics), 
             data_drift_score, performance_score)
        )
        
        conn.commit()
        conn.close()
        
        # Check for alerts
        self._check_for_alerts(model_id, metrics, data_drift_score, performance_score)
        
        return metric_id
        
    def _check_for_alerts(self, model_id: str, metrics: Dict, 
                         data_drift_score: float, performance_score: float):
        """Check if any alert thresholds have been triggered"""
        if data_drift_score and data_drift_score > 0.7:
            self.create_alert(model_id, "DATA_DRIFT", 
                             f"High data drift detected: {data_drift_score}")
            
        if performance_score and performance_score < 0.5:
            self.create_alert(model_id, "PERFORMANCE_DROP", 
                             f"Model performance below threshold: {performance_score}")
            
    def create_alert(self, model_id: str, alert_type: str, message: str) -> str:
        """Create a new alert"""
        alert_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?)",
            (alert_id, model_id, alert_type, message, timestamp, 0)  # 0 = not resolved
        )
        
        conn.commit()
        conn.close()
        
        print(f"ALERT [{alert_type}]: {message}")
        return alert_id
        
    def get_model_performance_history(self, model_id: str) -> List[Dict]:
        """Get historical performance for a model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT timestamp, metrics, data_drift_score, performance_score FROM model_metrics WHERE model_id = ? ORDER BY timestamp",
            (model_id,)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "metrics": json.loads(row[1]),
                "data_drift_score": row[2],
                "performance_score": row[3]
            }
            for row in results
        ]
        
    def get_active_alerts(self) -> List[Dict]:
        """Get all unresolved alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, model_id, alert_type, message, timestamp FROM alerts WHERE resolved = 0"
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "model_id": row[1],
                "alert_type": row[2],
                "message": row[3],
                "timestamp": row[4]
            }
            for row in results
        ]

# config.py
class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            return {}
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def save_config(self):
        """Save current configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if k not in value:
                return default
            value = value[k]
            
        return value
        
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        self.save_config()

# Usage example
if __name__ == "__main__":
    # Define pipeline steps
    def load_data(data_path):
        import pandas as pd
        data = pd.read_csv(data_path)
        return {"data": Artifact("raw_data", data, {"rows": len(data)})}
    
    def preprocess(data):
        # Perform preprocessing
        processed_data = data.dropna()
        return {"processed_data": Artifact("processed_data", processed_data)}
    
    def train_model(processed_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Simple example - replace with actual model training
        X = processed_data.drop("target", axis=1)
        y = processed_data["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        return {
            "model": Artifact("rf_model", model, {"accuracy": accuracy}),
            "metrics": {"accuracy": accuracy}
        }
    
    def evaluate_model(model, processed_data):
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Get test data
        X = processed_data.drop("target", axis=1)
        y = processed_data["target"]
        
        # Evaluate
        y_pred = model.data.predict(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted')
        }
        
        return {"evaluation": Artifact("model_evaluation", metrics)}
    
    # Create pipeline
    pipeline = Pipeline("example_ml_pipeline", "0.1.0")
    
    pipeline.add_step(PipelineStep(
        "load_data", 
        load_data, 
        inputs=["data_path"], 
        outputs=["data"]
    ))
    
    pipeline.add_step(PipelineStep(
        "preprocess", 
        preprocess, 
        inputs=["data"], 
        outputs=["processed_data"]
    ))
    
    pipeline.add_step(PipelineStep(
        "train_model", 
        train_model, 
        inputs=["processed_data"], 
        outputs=["model", "metrics"]
    ))
    
    pipeline.add_step(PipelineStep(
        "evaluate_model", 
        evaluate_model, 
        inputs=["model", "processed_data"], 
        outputs=["evaluation"]
    ))
    
    # Setup model registry and deployment
    model_registry = ModelRegistry(pipeline.db_path)
    deployer = Deployment(model_registry)
    monitor = ModelMonitor(model_registry)
    
    # Run pipeline
    run_id = pipeline.run({"data_path": "data/sample.csv"})
    
    # Register and deploy model
    model_id = model_registry.register_model(
        run_id, 
        "RandomForest_Classifier", 
        pipeline.steps[2].outputs["model"]
    )
    
    # Promote to production
    model_registry.promote_to_production(model_id)
    
    # Deploy model
    deployment_id = deployer.deploy_model(model_id)
    
    # Monitor model
    monitor.log_metrics(
        model_id,
        {"accuracy": 0.92, "latency_ms": 15},
        data_drift_score=0.2,
        performance_score=0.9
    )
