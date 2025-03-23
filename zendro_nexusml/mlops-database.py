# db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    runs = relationship("Run", back_populates="experiment")
    
    def __repr__(self):
        return f"<Experiment(name='{self.name}')>"


class Run(Base):
    __tablename__ = 'runs'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, unique=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    status = Column(String, default='created')  # created, running, completed, failed
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    config = Column(JSON)  # Store configuration as JSON
    
    # Relationships
    experiment = relationship("Experiment", back_populates="runs")
    metrics = relationship("Metric", back_populates="run")
    artifacts = relationship("Artifact", back_populates="run")
    
    def __repr__(self):
        return f"<Run(run_id='{self.run_id}', status='{self.status}')>"


class Metric(Base):
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="metrics")
    
    def __repr__(self):
        return f"<Metric(name='{self.name}', value={self.value})>"


class Artifact(Base):
    __tablename__ = 'artifacts'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # model, dataset, plot, etc.
    path = Column(String, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="artifacts")
    
    def __repr__(self):
        return f"<Artifact(name='{self.name}', type='{self.type}')>"


class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    artifact_id = Column(Integer, ForeignKey('artifacts.id'))
    status = Column(String, default='registered')  # registered, production, archived
    performance = Column(JSON)  # Store performance metrics as JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<Model(name='{self.name}', version='{self.version}', status='{self.status}')>"


class Deployment(Base):
    __tablename__ = 'deployments'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    environment = Column(String, nullable=False)  # dev, staging, production
    status = Column(String, nullable=False)  # pending, active, failed, rolled_back
    deployed_at = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)
    
    def __repr__(self):
        return f"<Deployment(environment='{self.environment}', status='{self.status}')>"


# db/operations.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any, List, Optional
import json
import datetime
from db.models import Base, Experiment, Run, Metric, Artifact, Model, Deployment

class DatabaseManager:
    def __init__(self, db_url: str):
        """Initialize database connection."""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_experiment(self, name: str, description: Optional[str] = None) -> Experiment:
        """Create a new experiment."""
        session = self.Session()
        experiment = Experiment(name=name, description=description)
        session.add(experiment)
        session.commit()
        experiment_id = experiment.id
        session.close()
        return experiment
    
    def create_run(self, experiment_id: int, run_id: str, config: Dict[str, Any]) -> Run:
        """Create a new run for an experiment."""
        session = self.Session()
        run = Run(
            run_id=run_id,
            experiment_id=experiment_id,
            status='created',
            start_time=datetime.datetime.utcnow(),
            config=config
        )
        session.add(run)
        session.commit()
        run_id = run.id
        session.close()
        return run
    
    def update_run_status(self, run_id: str, status: str) -> None:
        """Update the status of a run."""
        session = self.Session()
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run:
            run.status = status
            if status == 'completed' or status == 'failed':
                run.end_time = datetime.datetime.utcnow()
            session.commit()
        session.close()
    
    def log_metric(self, run_id: str, name: str, value: float) -> None:
        """Log a metric for a run."""
        session = self.Session()
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run:
            metric = Metric(
                run_id=run.id,
                name=name,
                value=value,
                timestamp=datetime.datetime.utcnow()
            )
            session.add(metric)
            session.commit()
        session.close()
    
    def log_artifact(self, run_id: str, name: str, artifact_type: str, 
                    path: str, metadata: Dict[str, Any]) -> Optional[Artifact]:
        """Log an artifact for a run."""
        session = self.Session()
        run = session.query(Run).filter(Run.run_id == run_id).first()
        artifact = None
        if run:
            artifact = Artifact(
                run_id=run.id,
                name=name,
                type=artifact_type,
                path=path,
                metadata=metadata,
                created_at=datetime.datetime.utcnow()
            )
            session.add(artifact)
            session.commit()
            artifact_id = artifact.id
        session.close()
        return artifact
    
    def register_model(self, name: str, version: str, artifact_id: int,
                      performance: Dict[str, float]) -> Model:
        """Register a new model version."""
        session = self.Session()
        model = Model(
            name=name,
            version=version,
            artifact_id=artifact_id,
            status='registered',
            performance=performance,
            created_at=datetime.datetime.utcnow()
        )
        session.add(model)
        session.commit()
        model_id = model.id
        session.close()
        return model
    
    def promote_model_to_production(self, model_id: int) -> None:
        """Promote a model to production status."""
        session = self.Session()
        # First, set all current production models of the same name to archived
        model = session.query(Model).filter(Model.id == model_id).first()
        if model:
            production_models = session.query(Model).filter(
                Model.name == model.name,
                Model.status == 'production'
            ).all()
            for prod_model in production_models:
                prod_model.status = 'archived'
            
            # Set the new model to production
            model.status = 'production'
            session.commit()
        session.close()
    
    def create_deployment(self, model_id: int, environment: str, 
                         endpoint: Optional[str] = None) -> Deployment:
        """Create a new deployment for a model."""
        session = self.Session()
        deployment = Deployment(
            model_id=model_id,
            environment=environment,
            status='pending',
            endpoint=endpoint
        )
        session.add(deployment)
        session.commit()
        deployment_id = deployment.id
        session.close()
        return deployment
    
    def update_deployment_status(self, deployment_id: int, status: str) -> None:
        """Update the status of a deployment."""
        session = self.Session()
        deployment = session.query(Deployment).filter(Deployment.id == deployment_id).first()
        if deployment:
            deployment.status = status
            session.commit()
        session.close()
    
    def get_latest_production_model(self, model_name: str) -> Optional[Model]:
        """Get the latest production model of a given name."""
        session = self.Session()
        model = session.query(Model).filter(
            Model.name == model_name,
            Model.status == 'production'
        ).order_by(Model.created_at.desc()).first()
        session.close()
        return model
    
    def get_experiment_metrics(self, experiment_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics for an experiment across all runs."""
        session = self.Session()
        results = {}
        
        # Get all runs for the experiment
        runs = session.query(Run).filter(Run.experiment_id == experiment_id).all()
        
        for run in runs:
            run_metrics = session.query(Metric).filter(Metric.run_id == run.id).all()
            metrics_list = []
            
            for metric in run_metrics:
                metrics_list.append({
                    'name': metric.name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat()
                })
            
            results[run.run_id] = metrics_list
        
        session.close()
        return results
    
    def compare_models(self, model_ids: List[int]) -> Dict[str, List[Dict[str, Any]]]:
        """Compare multiple models based on their performance metrics."""
        session = self.Session()
        results = {}
        
        for model_id in model_ids:
            model = session.query(Model).filter(Model.id == model_id).first()
            if model:
                results[f"{model.name}_v{model.version}"] = {
                    'performance': model.performance,
                    'created_at': model.created_at.isoformat(),
                    'status': model.status
                }
        
        session.close()
        return results
