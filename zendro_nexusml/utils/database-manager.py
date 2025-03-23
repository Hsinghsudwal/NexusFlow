import os
import logging
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import datetime

logger = logging.getLogger(__name__)
Base = declarative_base()

class DatabaseManager:
    """
    Manager for database operations
    Supports SQLite and PostgreSQL
    """
    
    def __init__(self, db_config):
        """Initialize the database manager with configuration"""
        self.db_config = db_config
        self.engine = None
        self.Session = None
        self.metadata = MetaData()
        
        if db_config["type"] == "sqlite":
            db_path = db_config["sqlite"]["path"]
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.engine = create_engine(f"sqlite:///{db_path}")
            logger.info(f"Connected to SQLite database at {db_path}")
        elif db_config["type"] == "postgres":
            host = db_config["postgres"]["host"]
            port = db_config["postgres"]["port"]
            user = db_config["postgres"]["user"]
            password = db_config["postgres"]["password"]
            database = db_config["postgres"]["database"]
            
            self.engine = create_engine(
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
            )
            logger.info(f"Connected to PostgreSQL database at {host}:{port}/{database}")
        else:
            raise ValueError(f"Unsupported database type: {db_config['type']}")
        
        self.Session = sessionmaker(bind=self.engine)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables"""
        # Pipeline runs table
        self.runs_table = Table(
            'pipeline_runs', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', String, unique=True, nullable=False),
            Column('start_time', DateTime, nullable=False),
            Column('end_time', DateTime, nullable=True),
            Column('status', String, nullable=False),
            Column('mode', String, nullable=False),
            Column('error', String, nullable=True),
        )
        
        # Model metrics table
        self.model_metrics_table = Table(
            'model_metrics', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('run_id', String, nullable=False),
            Column('model_version', String, nullable=False),
            Column('timestamp', DateTime, nullable=False),
            Column('accuracy', Float, nullable=True),
            Column('precision', Float, nullable=True),
            Column('recall', Float, nullable=True),
            Column('f1', Float, nullable=True),
            Column('roc_auc', Float, nullable=True),
            Column('additional_metrics', JSON, nullable=True),
        )
        
        # Monitoring metrics table
        self.monitoring_table = Table(
            'monitoring_metrics', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime, nullable=False),
            Column('model_version', String, nullable=False),
            Column('data_drift_score', Float, nullable=True),
            Column('performance_score', Float, nullable=True),
            Column('num_predictions', Integer, nullable=True),
            Column('avg_response_time', Float, nullable=True),
            Column('alerts', JSON, nullable=True),
        )
        
        # Create tables
        self.metadata.create_all(self.engine)
        logger.info("Database tables initialized")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
    
    def record_pipeline_run(self, run_id, start_time, status="running", mode="full"):
        """Record a new pipeline run"""
        with self.session_scope() as session:
            session.execute(
                self.runs_table.insert().values(
                    run_id=run_id,
                    start_time=start_time,
                    status=status,
                    mode=mode
                )
            )
        logger.info(f"Recorded pipeline run: {run_id}")
    
    def update_pipeline_run(self, run_id, end_time=None, status=None, error=None):
        """Update an existing pipeline run"""
        update_values = {}
        if end_time is not None:
            update_values["end_time"] = end_time
        if status is not None:
            update_values["status"] = status
        if error is not None:
            update_values["error"] = error
            
        if not update_values:
            logger.warning("No values provided for update")
            return
            
        with self.session_scope() as session:
            session.execute(
                self.runs_table.update()
                .where(self.runs_table.c.run_id == run_id)
                .values(**update_values)
            )
        logger.info(f"Updated pipeline run: {run_id}")
    
    def record_model_metrics(self, run_id, model_version, metrics):
        """Record model metrics"""
        with self.session_scope() as session:
            additional_metrics = {}
            standard_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
            
            # Separate standard metrics from additional ones
            for key, value in metrics.items():
                if key not in standard_metrics:
                    additional_metrics[key] = value
            
            # Insert the metrics
            session.execute(
                self.model_metrics_table.insert().values(
                    run_id=run_id,
                    model_version=model_version,
                    timestamp=datetime.datetime.now(),
                    accuracy=metrics.get('accuracy'),
                    precision=metrics.get('precision'),
                    recall=metrics.get('recall'),
                    f1=metrics.get('f1'),
                    roc_auc=metrics.get('roc_auc'),
                    additional_metrics=additional_metrics
                )
            )
        logger.info(f"Recorded model metrics for run: {run_id}")
    
    def record_monitoring_metrics(self, model_version, metrics):
        """Record monitoring metrics"""
        with self.session_scope() as session:
            session.execute(
                self.monitoring_table.insert().values(
                    timestamp=datetime.datetime.now(),
                    model_version=model_version,
                    data_drift_score=metrics.get('data_drift_score'),
                    performance_score=metrics.get('performance_score'),
                    num_predictions=metrics.get('num_predictions'),
                    avg_response_time=metrics.get('avg_response_time'),
                    alerts=metrics.get('alerts')
                )
            )
        logger.info(f"Recorded monitoring metrics for model: {model_version}")
    
    def get_latest_model_version(self):
        """Get the latest model version from the database"""
        with self.session_scope() as session:
            result = session.execute(
                "SELECT model_version FROM model_metrics "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            
            if result:
                return result[0]
            return None
    
    def get_model_metrics_history(self, model_version=None, limit=10):
        """Get historical model metrics"""
        query = "SELECT * FROM model_metrics "
        if model_version:
            query += f"WHERE model_version = '{model_version}' "
        query += "ORDER BY timestamp DESC "
        if limit:
            query += f"LIMIT {limit}"
            
        with self.session_scope() as session:
            result = session.execute(query).fetchall()
            # Convert to pandas DataFrame if needed
            df = pd.DataFrame(result)
            return df
    
    def get_monitoring_metrics_history(self, model_version=None, days=7):
        """Get historical monitoring metrics"""
        query = "SELECT * FROM monitoring_metrics "
        if model_version:
            query += f"WHERE model_version = '{model_version}' "
        if days:
            query += f"AND timestamp >= datetime('now', '-{days} days') "
        query += "ORDER BY timestamp DESC"
            
        with self.session_scope() as session:
            result = session.execute(query).fetchall()
            # Convert to pandas DataFrame if needed
            df = pd.DataFrame(result)
            return df
    
    def execute_query(self, query, params=None):
        """Execute a custom SQL query"""
        with self.session_scope() as session:
            result = session.execute(query, params or {}).fetchall()
            return result
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")
