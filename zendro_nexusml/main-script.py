import os
import logging
import argparse
from datetime import datetime
from prefect import flow, task, get_run_logger

from src.config import load_config
from src.constants import PIPELINE_STEPS
from src.pipelines.data_ingestion import DataIngestionPipeline
from src.pipelines.data_validation import DataValidationPipeline
from src.pipelines.data_preparation import DataPreparationPipeline
from src.pipelines.feature_engineering import FeatureEngineeringPipeline
from src.pipelines.training import ModelTrainingPipeline
from src.pipelines.evaluation import ModelEvaluationPipeline
from src.pipelines.deployment import ModelDeploymentPipeline
from src.pipelines.retraining import ModelRetrainingPipeline
from src.utils.artifact_manager import ArtifactManager
from src.utils.metadata import MetadataManager
from src.utils.db_utils import DatabaseManager
from src.utils.logging_utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

@task(name="load_configuration")
def load_configuration():
    """Load configuration from params.yaml"""
    logger.info("Loading configuration")
    return load_config()

@task(name="initialize_managers")
def initialize_managers(config):
    """Initialize artifact, metadata and database managers"""
    logger.info("Initializing managers")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_manager = ArtifactManager(
        base_path=os.path.join(config["artifacts"]["base_path"], run_id)
    )
    metadata_manager = MetadataManager(
        metadata_path=os.path.join(config["artifacts"]["metadata_path"], f"{run_id}.json")
    )
    db_manager = DatabaseManager(config["database"])
    
    return artifact_manager, metadata_manager, db_manager, run_id

@task(name="run_data_ingestion")
def run_data_ingestion(config, artifact_manager, metadata_manager):
    """Run data ingestion pipeline"""
    logger = get_run_logger()
    logger.info("Running data ingestion pipeline")
    
    pipeline = DataIngestionPipeline(config)
    data = pipeline.run()
    
    # Save artifacts and metadata
    artifact_path = artifact_manager.save_dataframe(data, "raw_data.csv")
    metadata_manager.log_step(
        step_name="data_ingestion",
        artifacts={"raw_data": artifact_path},
        metrics={"rows": len(data), "columns": len(data.columns)},
        parameters=config["data"]
    )
    
    return data

@task(name="run_data_validation")
def run_data_validation(data, config, artifact_manager, metadata_manager):
    """Run data validation pipeline"""
    logger = get_run_logger()
    logger.info("Running data validation pipeline")
    
    if not config["pipeline"]["run_data_validation"]:
        logger.info("Data validation step skipped as per config")
        return data
    
    pipeline = DataValidationPipeline(config)
    validated_data, validation_report = pipeline.run(data)
    
    # Save artifacts and metadata
    validation_report_path = artifact_manager.save_json(validation_report, "validation_report.json")
    metadata_manager.log_step(
        step_name="data_validation",
        artifacts={"validation_report": validation_report_path},
        metrics=validation_report["summary"],
        parameters={}
    )
    
    return validated_data

@task(name="run_data_preparation")
def run_data_preparation(data, config, artifact_manager, metadata_manager):
    """Run data preparation pipeline"""
    logger = get_run_logger()
    logger.info("Running data preparation pipeline")
    
    if not config["pipeline"]["run_data_preparation"]:
        logger.info("Data preparation step skipped as per config")
        return data
    
    pipeline = DataPreparationPipeline(config)
    processed_data, prep_info = pipeline.run(data)
    
    # Save artifacts and metadata
    processed_data_path = artifact_manager.save_dataframe(processed_data, "processed_data.csv")
    prep_info_path = artifact_manager.save_json(prep_info, "preparation_info.json")
    
    metadata_manager.log_step(
        step_name="data_preparation",
        artifacts={
            "processed_data": processed_data_path,
            "preparation_info": prep_info_path
        },
        metrics={"rows_after_processing": len(processed_data)},
        parameters=prep_info["parameters"]
    )
    
    return processed_data

@task(name="run_feature_engineering")
def run_feature_engineering(data, config, artifact_manager, metadata_manager):
    """Run feature engineering pipeline"""
    logger = get_run_logger()
    logger.info("Running feature engineering pipeline")
    
    if not config["pipeline"]["run_feature_engineering"]:
        logger.info("Feature engineering step skipped as per config")
        return data, None
    
    pipeline = FeatureEngineeringPipeline(config)
    features, feature_info, feature_pipeline = pipeline.run(data)
    
    # Save artifacts and metadata
    features_path = artifact_manager.save_dataframe(features, "features.csv")
    feature_info_path = artifact_manager.save_json(feature_info, "feature_info.json")
    feature_pipeline_path = artifact_manager.save_pickle(feature_pipeline, "feature_pipeline.pkl")
    
    metadata_manager.log_step(
        step_name="feature_engineering",
        artifacts={
            "features": features_path,
            "feature_info": feature_info_path,
            "feature_pipeline": feature_pipeline_path
        },
        metrics={"feature_count": len(features.columns) - 1},  # Exclude target
        parameters=config["features"]
    )
    
    return features, feature_pipeline

@task(name="run_model_training")
def run_model_training(features, config, artifact_manager, metadata_manager):
    """Run model training pipeline"""
    logger = get_run_logger()
    logger.info("Running model training pipeline")
    
    if not config["pipeline"]["run_training"]:
        logger.info("Model training step skipped as per config")
        return None, None
    
    pipeline = ModelTrainingPipeline(config)
    model, training_info = pipeline.run(features)
    
    # Save artifacts and metadata
    model_path = artifact_manager.save_pickle(model, "model.pkl")
    training_info_path = artifact_manager.save_json(training_info, "training_info.json")
    
    metadata_manager.log_step(
        step_name="model_training",
        artifacts={
            "model": model_path,
            "training_info": training_info_path
        },
        metrics=training_info["metrics"],
        parameters=config["model"]["hyperparameters"][config["model"]["algorithm"]]
    )
    
    return model, training_info

@task(name="run_model_evaluation")
def run_model_evaluation(model, features, feature_pipeline, config, artifact_manager, metadata_manager):
    """Run model evaluation pipeline"""
    logger = get_run_logger()
    logger.info("Running model evaluation pipeline")
    
    if not config["pipeline"]["run_evaluation"] or model is None:
        logger.info("Model evaluation step skipped as per config or missing model")
        return False
    
    pipeline = ModelEvaluationPipeline(config)
    evaluation_result = pipeline.run(model, features, feature_pipeline)
    
    # Save artifacts and metadata
    eval_result_path = artifact_manager.save_json(evaluation_result, "evaluation_result.json")
    
    is_production_ready = evaluation_result["metrics"]["accuracy"] >= config["evaluation"]["production_min_accuracy"]
    
    metadata_manager.log_step(
        step_name="model_evaluation",
        artifacts={"evaluation_result": eval_result_path},
        metrics=evaluation_result["metrics"],
        parameters={"is_production_ready": is_production_ready}
    )
    
    return is_production_ready

@task(name="run_model_deployment")
def run_model_deployment(is_production_ready, model, feature_pipeline, run_id, config, artifact_manager, metadata_manager):
    """Run model deployment pipeline"""
    logger = get_run_logger()
    logger.info("Running model deployment pipeline")
    
    if not config["pipeline"]["run_deployment"] or not is_production_ready:
        logger.info("Model deployment step skipped as per config or model not production ready")
        return
    
    deployment_pipeline = ModelDeploymentPipeline(config)
    deployment_result = deployment_pipeline.run(model, feature_pipeline, run_id)
    
    # Save artifacts and metadata
    deployment_result_path = artifact_manager.save_json(deployment_result, "deployment_result.json")
    
    metadata_manager.log_step(
        step_name="model_deployment",
        artifacts={"deployment_result": deployment_result_path},
        metrics={},
        parameters={"deployed_model_version": deployment_result["model_version"]}
    )

@flow(name="customer_churn_pipeline")
def run_pipeline(mode="full"):
    """Main pipeline flow"""
    logger = get_run_logger()
    logger.info(f"Starting pipeline in {mode} mode")
    
    # Load configuration
    config = load_configuration()
    
    # Initialize managers
    artifact_manager, metadata_manager, db_manager, run_id = initialize_managers(config)
    
    try:
        metadata_manager.start_run(run_id=run_id, mode=mode)
        
        if mode == "full" or mode == "training":
            # Data ingestion
            data = run_data_ingestion(config, artifact_manager, metadata_manager)
            
            # Data validation
            validated_data = run_data_validation(data, config, artifact_manager, metadata_manager)
            
            # Data preparation
            processed_data = run_data_preparation(validated_data, config, artifact_manager, metadata_manager)
            
            # Feature engineering
            features, feature_pipeline = run_feature_engineering(processed_data, config, artifact_manager, metadata_manager)
            
            # Model training
            model, training_info = run_model_training(features, config, artifact_manager, metadata_manager)
            
            # Model evaluation
            is_production_ready = run_model_evaluation(model, features, feature_pipeline, config, artifact_manager, metadata_manager)
            
            # Model deployment
            if mode == "full":
                run_model_deployment(is_production_ready, model, feature_pipeline, run_id, config, artifact_manager, metadata_manager)
        
        elif mode == "retraining":
            # Run retraining pipeline
            retraining_pipeline = ModelRetrainingPipeline(config)
            retraining_result = retraining_pipeline.run()
            
            # Log retraining results
            metadata_manager.log_step(
                step_name="model_retraining",
                artifacts={},
                metrics=retraining_result["metrics"],
                parameters=retraining_result["parameters"]
            )
            
        metadata_manager.end_run(status="completed")
        logger.info(f"Pipeline completed successfully in {mode} mode")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        metadata_manager.end_run(status="failed", error=str(e))
        raise
    
    finally:
        # Close database connections
        db_manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLOps pipeline for customer churn prediction")
    parser.add_argument("--mode", type=str, choices=["full", "training", "retraining"], default="full",
                        help="Pipeline execution mode")
    args = parser.parse_args()
    
    run_pipeline(mode=args.mode)





# main.py

from pipeline.data_ingestion import DataIngestion
from pipeline.preprocessing import Preprocessing
from pipeline.model_training import ModelTraining
from pipeline.evaluation import Evaluation
from pipeline.deployment import Deployment
from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from experiments.experiment_tracker import ExperimentTracker
from artifact_management.artifact_manager import ArtifactManager

# Define the pipeline steps
pipeline = [
    DataIngestion(),
    Preprocessing(),
    ModelTraining(),
    Evaluation(),
    Deployment()
]

# Initialize the orchestrator
orchestrator = PipelineOrchestrator(pipeline)

# Example configuration and data
config = {'data_source': 'data_source_url'}
data = "raw_data"

# Run the pipeline
orchestrator.execute(config, data)

# Optionally, track experiments and save models
experiment_tracker = ExperimentTracker()
experiment_tracker.log_experiment("experiment_001", config, {'accuracy': 0.95}, "best_model")

artifact_manager = ArtifactManager("/path/to/artifacts")
artifact_manager.save_artifact("best_model", "model_data")
