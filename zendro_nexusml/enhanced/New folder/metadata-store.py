# mlops_framework/metadata_store.py

import os
import json
import sqlite3
import pickle
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataStore:
    """
    Singleton class to manage metadata for pipelines and steps.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetadataStore, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the metadata store with database connection."""
        self.db_path = os.environ.get("METADATA_DB_PATH", "metadata.db")
        self._create_tables()
        logger.info(f"Initialized metadata store at {self.db_path}")
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create pipeline runs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            pipeline_id TEXT,
            pipeline_name TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            error TEXT
        )
        ''')
        
        # Create step runs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS step_runs (
            step_run_id TEXT PRIMARY KEY,
            run_id TEXT,
            step_id TEXT,
            step_name TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            error TEXT,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
        )
        ''')
        
        # Create artifacts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            step_run_id TEXT,
            name TEXT,
            path TEXT,
            metadata TEXT,
            FOREIGN KEY (step_run_id) REFERENCES step_runs (step_run_id)
        )
        ''')
        
        # Create cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            data BLOB,
            timestamp TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_pipeline_run(self, pipeline_id: str, pipeline_name: str, start_time: datetime) -> str:
        """
        Register a new pipeline run.
        
        Args:
            pipeline_id: ID of the pipeline
            pipeline_name: Name of the pipeline
            start_time: Start time of the run
            
        Returns:
            ID of the run
        """
        import uuid
        run_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO pipeline_runs (run_id, pipeline_id, pipeline_name, start_time, status) VALUES (?, ?, ?, ?, ?)",
            (run_id, pipeline_id, pipeline_name, start_time.isoformat(), "running")
        )
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def update_pipeline_run(self, run_id: str, status: str, end_time: datetime, error: Optional[str] = None):
        """
        Update an existing pipeline run.
        
        Args:
            run_id: ID of the run
            status: New status
            end_time: End time of the run
            error: Optional error message
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE pipeline_runs SET status = ?, end_time = ?, error = ? WHERE run_id = ?",
            (status, end_time.isoformat(), error, run_id)
        )
        
        conn.commit()
        conn.close()
    
    def register_step_run(self, run_id: str, step_id: str, step_name: str, start_time: datetime) -> str:
        """
        Register a new step run.
        
        Args:
            run_id: ID of the pipeline run
            step_id: ID of the step
            step_name: Name of the step
            start_time: Start time of the run
            
        Returns:
            ID of the step run
        """
        import uuid
        step_run_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO step_runs (step_run_id, run_id, step_id, step_name, start_time, status) VALUES (?, ?, ?, ?, ?, ?)",
            (step_run_id, run_id, step_id, step_name, start_time.isoformat(), "running")
        )
        
        conn.commit()
        conn.close()
        
        return step_run_id
    
    def update_step_run(self, step_run_id: str, status: str, end_time: datetime, error: Optional[str] = None):
        """
        Update an existing step run.
        
        Args:
            step_run_id: ID of the step run
            status: New status
            end_time: End time of the run
            error: Optional error message
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE step_runs SET status = ?, end_time = ?, error = ? WHERE step_run_id = ?",
            (status, end_time.isoformat(), error, step_run_id)
        )
        
        conn.commit()
        conn.close()
    
    def cache_result(self, cache_key: str, result: Any):
        """
        Cache a step result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the result
        serialized = pickle.dumps(result)
        
        cursor.execute(
            "INSERT OR REPLACE INTO cache (cache_key, data, timestamp) VALUES (?, ?, ?)",
            (cache_key, serialized, datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Get a cached result.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT data FROM cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return pickle.loads(row[0])
        return None
    
    def get_pipeline_runs(self, pipeline_id: Optional[str] = None) -> List[Dict]:
        """
        Get all pipeline runs, optionally filtered by pipeline ID.
        
        Args:
            pipeline_id: Optional pipeline ID to filter by
            
        Returns:
            List of pipeline runs
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if pipeline_id:
            cursor.execute("SELECT * FROM pipeline_runs WHERE pipeline_id = ?", (pipeline_id,))
        else:
            cursor.execute("SELECT * FROM pipeline_runs")
        
        runs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return runs
    
    def get_step_runs(self, run_id: str) -> List[Dict]:
        """
        Get all step runs for a pipeline run.
        
        Args:
            run_id: Pipeline run ID
            
        Returns:
            List of step runs
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM step_runs WHERE run_id = ?", (run_id,))
        
        runs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return runs
