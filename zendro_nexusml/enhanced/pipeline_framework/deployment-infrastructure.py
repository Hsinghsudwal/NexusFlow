# deployment.py
import os
import yaml
import json
import time
import uuid
import sqlite3
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineTracker:
    """Tracks pipeline runs and maintains execution history."""
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize the pipeline tracker with a database path."""
        self.db_path = Path(db_path)
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite database for tracking pipeline runs."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            pipeline_name TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            environment TEXT,
            params TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS node_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            node_name TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            inputs TEXT,
            outputs TEXT,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            name TEXT,
            value REAL,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def start_pipeline_run(
        self, 
        pipeline_name: str, 
        params: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking a new pipeline run."""
        run_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO pipeline_runs 
            (run_id, pipeline_name, start_time, status, environment, params) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                run_id, 
                pipeline_name, 
                now, 
                'RUNNING', 
                json.dumps(environment or {}),
                json.dumps(params or {})
            )
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Started tracking pipeline run {run_id}")
        return run_id
    
    def finish_pipeline_run(self, run_id: str, status: str = 'COMPLETED'):
        """Mark a pipeline run as completed."""
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            UPDATE pipeline_runs 
            SET status = ?, end_time = ? 
            WHERE run_id = ?
            ''',
            (status, now, run_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Completed pipeline run {run_id} with status {status}")
    
    def start_node_run(self, run_id: str, node_name: str, inputs: Optional[Dict[str, Any]] = None):
        """Start tracking a node execution within a pipeline run."""
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO node_runs 
            (run_id, node_name, start_time, status, inputs) 
            VALUES (?, ?, ?, ?, ?)
            ''',
            (run_id, node_name, now, 'RUNNING', json.dumps(inputs or {}))
        )
        
        conn.commit()
        conn.close()
    
    def finish_node_run(
        self, 
        run_id: str, 
        node_name: str, 
        status: str = 'COMPLETED', 
        outputs: Optional[Dict[str, Any]] = None
    ):
        """Mark a node execution as completed."""
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            UPDATE node_runs 
            SET status = ?, end_time = ?, outputs = ? 
            WHERE run_id = ? AND node_name = ? AND end_time IS NULL
            ''',
            (status, now, json.dumps(outputs or {}), run_id, node_name)
        )
        
        conn.commit()
        conn.close()
    
    def log_metric(self, run_id: str, name: str, value: float):
        """Log a metric associated with a pipeline run."""
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO metrics 
            (run_id, name, value, timestamp) 
            VALUES (?, ?, ?, ?)
            ''',
            (run_id, name, value, now)
        )
        
        conn.commit()
        conn.close()
    
    def get_pipeline_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent pipeline runs."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            SELECT * FROM pipeline_runs 
            ORDER BY start_time DESC 
            LIMIT ?
            ''',
            (limit,)
        )
        
        runs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return runs
    
    def get_pipeline_run(self, run_id: str) -> Dict[str, Any]:
        """Get details of a specific pipeline run including nodes and metrics."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get pipeline run details
        cursor.execute('SELECT * FROM pipeline_runs WHERE run_id = ?', (run_id,))
        pipeline_run = dict(cursor.fetchone())
        
        # Get node runs for this pipeline
        cursor.execute('SELECT * FROM node_runs WHERE run_id = ?', (run_id,))
        node_runs = [dict(row) for row in cursor.fetchall()]
        
        # Get metrics for this pipeline
        cursor.execute('SELECT * FROM metrics WHERE run_id = ?', (run_id,))
        metrics = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        pipeline_run['node_runs'] = node_runs
        pipeline_run['metrics'] = metrics
        
        return pipeline_run

class ModelRegistry:
    """Registry for tracking ML models and their metadata."""
    
    def __init__(self, registry_path: Union[str, Path]):
        """Initialize the model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_db_path = self.registry_path / 'models.db'
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite database for the model registry."""
        conn = sqlite3.connect(str(self.models_db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            name TEXT,
            version TEXT,
            created_at TEXT,
            run_id TEXT,
            path TEXT,
            metadata TEXT,
            is_production BOOLEAN DEFAULT 0
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_model(
        self, 
        name: str, 
        model_path: Union[str, Path], 
        run_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a model in the registry."""
        model_id = str(uuid.uuid4())
        model_path = Path(model_path)
        
        # Determine the next version number
        conn = sqlite3.connect(str(self.models_db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT MAX(CAST(version AS INTEGER)) FROM models WHERE name = ?', 
            (name,)
        )
        result = cursor.fetchone()[0]
        version = str(1 if result is None else result + 1)
        
        # Create target directory for the model
        target_dir = self.registry_path / name / version
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to registry
        import shutil
        if model_path.is_file():
            target_path = target_dir / model_path.name
            shutil.copy2(model_path, target_path)
        else:
            target_path = target_dir
            shutil.copytree(model_path, target_path, dirs_exist_ok=True)
        
        # Register model in database
        now = datetime.datetime.now().isoformat()
        cursor.execute(
            '''
            INSERT INTO models 
            (model_id, name, version, created_at, run_id, path, metadata) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                model_id, 
                name, 
                version, 
                now, 
                run_id or '', 
                str(target_dir), 
                json.dumps(metadata or {})
            )
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Registered model {name} version {version} with ID {model_id}")
        return model_id
    
    def promote_to_production(self, model_id: str):
        """Promote a model to production status."""
        conn = sqlite3.connect(str(self.models_db_path))
        cursor = conn.cursor()
        
        # Reset any current production models for this model type
        cursor.execute(
            'SELECT name FROM models WHERE model_id = ?', 
            (model_id,)
        )
        name = cursor.fetchone()[0]
        
        cursor.execute(
            'UPDATE models SET is_production = 0 WHERE name = ?', 
            (name,)
        )
        
        # Set this model as production
        cursor.execute(
            'UPDATE models SET is_production = 1 WHERE model_id = ?', 
            (model_id,)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Promoted model {model_id} to production status")
    
    def get_production_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the current production model for a given model type."""
        conn = sqlite3.connect(str(self.models_db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM models WHERE name = ? AND is_production = 1', 
            (name,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by ID."""
        conn = sqlite3.connect(str(self.models_db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    def get_models(self, name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get list of models, optionally filtered by name."""
        conn = sqlite3.connect(str(self.models_db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if name:
            cursor.execute(
                'SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT ?',
                (name, limit)
            )
        else:
            cursor.execute(
                'SELECT * FROM models ORDER BY created_at DESC LIMIT ?',
                (limit,)
            )
        
        models = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return models

# Extend the Pipeline class to integrate with tracking
def track_pipeline(original_run):
    """Decorator to add tracking to pipeline runs."""
    def tracked_run(self, catalog, **kwargs):
        # Initialize tracker
        tracker = PipelineTracker(Path(catalog.base_path.parent) / 'tracker.db')
        
        # Start pipeline tracking
        run_id = tracker.start_pipeline_run(
            self.name,
            params=kwargs,
            environment={
                'python_version': os.environ.get('PYTHON_VERSION', ''),
                'user': os.environ.get('USER', ''),
                'host': os.environ.get('HOSTNAME', ''),
            }
        )
        
        try:
            # Sort nodes to determine execution order
            execution_order = self._sort_nodes()
            filtered_nodes = []
            
            # Apply node filters
            if kwargs.get('only_nodes'):
                only_node_names = set(kwargs['only_nodes'])
                filtered_nodes = [n for n in execution_order if n.name in only_node_names]
            elif kwargs.get('from_nodes') or kwargs.get('to_nodes'):
                if kwargs.get('from_nodes'):
                    include_after = False
                    for node in execution_order:
                        if node.name in kwargs['from_nodes']:
                            include_after = True
                        if include_after:
                            filtered_nodes.append(node)
                else:
                    filtered_nodes = execution_order.copy()
                
                if kwargs.get('to_nodes'):
                    temp_nodes = []
                    include_before = True
                    for node in filtered_nodes:
                        if include_before:
                            temp_nodes.append(node)
                        if node.name in kwargs['to_nodes']:
                            include_before = False
                    filtered_nodes = temp_nodes
            else:
                filtered_nodes = execution_order
            
            if kwargs.get('tags'):
                tag_set = set(kwargs['tags'])
                filtered_nodes = [n for n in filtered_nodes if n.tags.intersection(tag_set)]
            
            # Run pipeline with tracking for each node
            results = {}
            for node in filtered_nodes:
                # Track node start
                tracker.start_node_run(run_id, node.name, inputs={k: 'data_loaded' for k in node.inputs})
                
                try:
                    # Load inputs from catalog
                    input_data = {}
                    for input_name in node.inputs:
                        input_data[input_name] = catalog.load(input_name)
                    
                    # Execute function
                    logger.info(f"Running node '{node.name}'")
                    start_time = time.time()
                    
                    if node.inputs:
                        outputs = node.func(**input_data)
                    else:
                        outputs = node.func()
                    
                    # Handle different output structures
                    if len(node.outputs) == 1:
                        outputs = {node.outputs[0]: outputs}
                    elif not isinstance(outputs, tuple) and len(node.outputs) > 1:
                        raise ValueError(f"Expected {len(node.outputs)} outputs but got a single value")
                    elif isinstance(outputs, tuple) and len(outputs) != len(node.outputs):
                        raise ValueError(f"Expected {len(node.outputs)} outputs but got {len(outputs)}")
                    else:
                        outputs = dict(zip(node.outputs, outputs))
                    
                    duration = time.time() - start_time
                    
                    # Save outputs to catalog
                    for name, data in outputs.items():
                        catalog.save(name, data, metadata={'node': node.name, 'duration': duration})
                    
                    # Update results
                    results.update(outputs)
                    
                    # Track node completion
                    tracker.finish_node_run(
                        run_id, 
                        node.name, 
                        status='COMPLETED',
                        outputs={k: 'data_saved' for k in node.outputs}
                    )
                    
                except Exception as e:
                    # Track node failure
                    tracker.finish_node_run(run_id, node.name, status='FAILED')
                    # Mark pipeline as failed
                    tracker.finish_pipeline_run(run_id, status='FAILED')
                    # Re-raise exception
                    raise e
            
            # Mark pipeline as completed
            tracker.finish_pipeline_run(run_id)
            return results
            
        except Exception as e:
            # Mark pipeline as failed if not already done
            try:
                tracker.finish_pipeline_run(run_id, status='FAILED')
            except:
                pass
            # Re-raise the exception
            raise e
    
    return tracked_run

# Update existing Pipeline class to use tracking
from functools import wraps

def inject_tracking(pipeline_class):
    """Inject tracking functionality into Pipeline class."""
    original_run = pipeline_class.run
    
    @wraps(original_run)
    def tracked_run(self, catalog, **kwargs):
        return track_pipeline(original_run)(self, catalog, **kwargs)
    
    pipeline_class.run = tracked_run
    return pipeline_class

# Registry extensions for the ProjectContext
def extend_project_context(context_class):
    """Extend ProjectContext with model registry and tracking."""
    
    original_init = context_class.__init__
    
    def extended_init(self, project_path):
        original_init(self, project_path)
        self.tracker = PipelineTracker(Path(project_path) / 'tracking' / 'tracker.db')
        self.model_registry = ModelRegistry(Path(project_path) / 'models')
    
    context_class.__init__ = extended_init
    return context_class
