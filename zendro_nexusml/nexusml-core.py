# nexusml/core/artifact.py
import os
import json
import hashlib
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

class Artifact:
    """Base class for all artifacts in NexusML."""
    
    def __init__(self, name: str, data: Any, metadata: Optional[Dict] = None):
        self.name = name
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.artifact_id = self._generate_id()
        
    def _generate_id(self) -> str:
        """Generate a unique ID for the artifact."""
        content_hash = hashlib.md5(pickle.dumps(self.data)).hexdigest()
        return f"{self.name}-{content_hash[:10]}"
    
    def save(self, path: str) -> str:
        """Save the artifact to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save the data
        data_path = os.path.join(path, f"{self.name}.pickle")
        with open(data_path, "wb") as f:
            pickle.dump(self.data, f)
        
        # Save the metadata
        metadata = {
            "name": self.name,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            **self.metadata
        }
        metadata_path = os.path.join(path, f"{self.name}.metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return data_path
    
    @classmethod
    def load(cls, path: str, metadata_path: Optional[str] = None) -> 'Artifact':
        """Load an artifact from disk."""
        if metadata_path is None:
            metadata_path = path.replace(".pickle", ".metadata.json")
        
        # Load the data
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Load the metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        name = metadata.pop("name")
        artifact = cls(name=name, data=data, metadata=metadata)
        artifact.created_at = metadata.pop("created_at")
        artifact.artifact_id = metadata.pop("artifact_id")
        
        return artifact


# nexusml/core/metadata.py
import os
import json
import sqlite3
from typing import Dict, List, Optional, Any

class MetadataStore:
    """Store and retrieve metadata for pipelines, steps, and artifacts."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipelines (
            pipeline_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            metadata TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            step_id TEXT PRIMARY KEY,
            pipeline_id TEXT,
            name TEXT,
            created_at TEXT,
            metadata TEXT,
            FOREIGN KEY (pipeline_id) REFERENCES pipelines (pipeline_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            step_id TEXT,
            name TEXT,
            created_at TEXT,
            path TEXT,
            metadata TEXT,
            FOREIGN KEY (step_id) REFERENCES steps (step_id)
        )
        """)
        
        conn.commit()
        conn.close()
    
    def save_pipeline_metadata(self, pipeline_id: str, name: str, created_at: str, metadata: Dict):
        """Save pipeline metadata to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO pipelines VALUES (?, ?, ?, ?)",
            (pipeline_id, name, created_at, json.dumps(metadata))
        )
        
        conn.commit()
        conn.close()
    
    def save_step_metadata(self, step_id: str, pipeline_id: str, name: str, created_at: str, metadata: Dict):
        """Save step metadata to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?, ?)",
            (step_id, pipeline_id, name, created_at, json.dumps(metadata))
        )
        
        conn.commit()
        conn.close()
    
    def save_artifact_metadata(self, artifact_id: str, step_id: str, name: str, created_at: str, path: str, metadata: Dict):
        """Save artifact metadata to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
            (artifact_id, step_id, name, created_at, path, json.dumps(metadata))
        )
        
        conn.commit()
        conn.close()
    
    def get_pipeline_metadata(self, pipeline_id: str) -> Optional[Dict]:
        """Get pipeline metadata from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM pipelines WHERE pipeline_id = ?", (pipeline_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            return None
        
        return {
            "pipeline_id": row[0],
            "name": row[1],
            "created_at": row[2],
            "metadata": json.loads(row[3])
        }
    
    def get_step_metadata(self, step_id: str) -> Optional[Dict]:
        """Get step metadata from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM steps WHERE step_id = ?", (step_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            return None
        
        return {
            "step_id": row[0],
            "pipeline_id": row[1],
            "name": row[2],
            "created_at": row[3],
            "metadata": json.loads(row[4])
        }
    
    def get_artifact_metadata(self, artifact_id: str) -> Optional[Dict]:
        """Get artifact metadata from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            return None
        
        return {
            "artifact_id": row[0],
            "step_id": row[1],
            "name": row[2],
            "created_at": row[3],
            "path": row[4],
            "metadata": json.loads(row[5])
        }
    
    def list_pipelines(self) -> List[Dict]:
        """List all pipelines in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM pipelines")
        rows = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                "pipeline_id": row[0],
                "name": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3])
            }
            for row in rows
        ]
    
    def list_pipeline_steps(self, pipeline_id: str) -> List[Dict]:
        """List all steps for a pipeline in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM steps WHERE pipeline_id = ?", (pipeline_id,))
        rows = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                "step_id": row[0],
                "pipeline_id": row[1],
                "name": row[2],
                "created_at": row[3],
                "metadata": json.loads(row[4])
            }
            for row in rows
        ]
    
    def list_step_artifacts(self, step_id: str) -> List[Dict]:
        """List all artifacts for a step in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM artifacts WHERE step_id = ?", (step_id,))
        rows = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                "artifact_id": row[0],
                "step_id": row[1],
                "name": row[2],
                "created_at": row[3],
                "path": row[4],
                "metadata": json.loads(row[5])
            }
            for row in rows
        ]


# nexusml/core/context.py
import os
import json
from typing import Dict, Any, Optional

class ExecutionContext:
    """Execution context for a pipeline run."""
    
    def __init__(self, pipeline_id: str, run_id: str, working_dir: str):
        self.pipeline_id = pipeline_id
        self.run_id = run_id
        self.working_dir = working_dir
        self.step_outputs = {}
        self._ensure_working_dir()
    
    def _ensure_working_dir(self):
        """Ensure that the working directory exists."""
        os.makedirs(self.working_dir, exist_ok=True)
    
    def get_artifact_dir(self, step_id: str) -> str:
        """Get the directory for step artifacts."""
        artifact_dir = os.path.join(self.working_dir, "artifacts", step_id)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir
    
    def save_step_output(self, step_id: str, outputs: Dict[str, Any]):
        """Save step outputs to the context."""
        self.step_outputs[step_id] = outputs
    
    def get_step_output(self, step_id: str, output_name: str) -> Optional[Any]:
        """Get a step output from the context."""
        step_outputs = self.step_outputs.get(step_id)
        if step_outputs is None:
            return None
        return step_outputs.get(output_name)
    
    def save_state(self):
        """Save the context state to disk."""
        state_path = os.path.join(self.working_dir, "context_state.json")
        
        # We can't directly serialize the outputs, so just save the structure
        serializable_outputs = {
            step_id: list(outputs.keys())
            for step_id, outputs in self.step_outputs.items()
        }
        
        state = {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "step_outputs": serializable_outputs
        }
        
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, working_dir: str) -> 'ExecutionContext':
        """Load the context state from disk."""
        state_path = os.path.join(working_dir, "context_state.json")
        
        with open(state_path, "r") as f:
            state = json.load(f)
        
        context = cls(
            pipeline_id=state["pipeline_id"],
            run_id=state["run_id"],
            working_dir=working_dir
        )
        
        # Note: This doesn't actually load the step outputs,
        # just the structure. The actual outputs would need
        # to be loaded separately.
        
        return context
