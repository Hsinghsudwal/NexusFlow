import time
import json
import logging
import mlflow
import numpy as np
from typing import Dict, Tuple, Any, Optional
from mlflow.tracking import MlflowClient
from datetime import datetime
from src.utils.config_manager import ConfigManager
from utils.logger import logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)


def load_models(
    client,
    model_name: str,
    stage_staging,
    stage_prod,
) -> Tuple[Any, Optional[Any]]:
    """Load models from two different stages for comparison"""
    logger.info(f"Loading models for {model_name}: {stage_staging} and {stage_prod}")

    # Get model from staging
    staging_versions = client.get_latest_versions(model_name, stages=[stage_staging])
    if not staging_versions:
        raise ValueError(
            f"No versions found for model: {model_name} in {stage_staging}"
        )

    model_uri_stage = f"models:/{model_name}/{stage_staging}"
    model_stage = mlflow.pyfunc.load_model(model_uri_stage)

    # Get model from production (if exists)
    prod_versions = client.get_latest_versions(model_name, stages=[stage_prod])
    if not prod_versions:
        logger.warning(f"No model found in {stage_prod} stage")
        model_prod = None
    else:
        model_uri_prod = f"models:/{model_name}/{stage_prod}"
        model_prod = mlflow.pyfunc.load_model(model_uri_prod)

    return model_stage, model_prod


def evaluate_model(config, model, X_test, y_test) -> Dict[str, float]:
    """Evaluate a model using appropriate metrics based on problem type"""
    problem_type = config.get("base", {}).get("problem_type", "regression")
    metrics = {}

    try:
        predictions = model.predict(X_test)

        if problem_type == "classification":
            # For binary classification
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["precision"] = precision_score(
                y_test, predictions, average="weighted"
            )
            metrics["recall"] = recall_score(y_test, predictions, average="weighted")
            metrics["f1"] = f1_score(y_test, predictions, average="weighted")

            # Try to get probability predictions for ROC AUC if available
            if hasattr(model, "predict_proba"):
                proba_predictions = model.predict_proba(X_test)
                # For binary classification
                if proba_predictions.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, proba_predictions[:, 1])
                # For multiclass
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_test, proba_predictions, multi_class="ovr"
                    )

        else:  # regression
            metrics["rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics["mae"] = mean_absolute_error(y_test, predictions)
            metrics["r2"] = r2_score(y_test, predictions)
            metrics["explained_variance"] = explained_variance_score(
                y_test, predictions
            )

        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {}


def compare_metrics(
    config,
    staging_metrics: Dict[str, float],
    production_metrics: Dict[str, float],
) -> Tuple[bool, Dict[str, float]]:
    """Compare metrics between staging and production models"""
    problem_type = config.get("base", {}).get("problem_type", "regression")

    # Get the primary metric for comparison
    if problem_type == "classification":
        primary_metric = config.get("promotion_thresholds", {}).get("primary_metric", "f1")
        # Define which metrics should be higher (better)
        higher_is_better = {
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1": True,
            "roc_auc": True,
        }
    else:  # regression
        primary_metric = config.get("promotion_thresholds", {}).get("primary_metric", "rmse")
        # Define which metrics should be higher (better)
        higher_is_better = {
            "rmse": False,
            "mae": False,  # Lower is better for error metrics
            "r2": True,
            "explained_variance": True,  # Higher is better
        }

    # Calculate the difference in metrics
    diff_metrics = {}
    for metric in staging_metrics:
        if metric in production_metrics:
            diff = staging_metrics[metric] - production_metrics[metric]
            diff_metrics[metric] = diff

    # Check if the primary metric improved by the required threshold
    minimum_improvement = config.get("promotion_thresholds", {}).get(
        "minimum_improvement", 0.01  
    )  # 1% improvement by default. Test try just 0

    if primary_metric not in diff_metrics:
        return False, diff_metrics

    is_better = False
    if higher_is_better.get(primary_metric, True):
        # Higher is better (e.g., accuracy, f1, r2)
        is_better = diff_metrics[primary_metric] >= minimum_improvement
    else:
        # Lower is better (e.g., rmse, mae)
        is_better = diff_metrics[primary_metric] <= -minimum_improvement

    # Check secondary metrics - all must be at least not worse than a certain tolerance
    secondary_tolerance = config.get("promotion_thresholds", {}).get(
        "secondary_metric_tolerance", 0.05
    )  # 5% tolerance

    for metric, diff in diff_metrics.items():
        if metric == primary_metric:
            continue  # Already checked

        if higher_is_better.get(metric, True):
            # For metrics where higher is better
            if diff < -secondary_tolerance:
                is_better = False
                break
        else:
            # For metrics where lower is better
            if diff > secondary_tolerance:
                is_better = False
                break

    return is_better, diff_metrics


def promote_to_production(config_file, results: Dict) -> bool:
    """Promote model to production based on evaluation results"""
    try:
        config = ConfigManager.load_file(config_file)

        # Get MLflow configuration
        tracking_uri = config.get("mlflow_config", {}).get("mlflow_tracking_uri")
        remote_server_uri = config.get("mlflow_config", {}).get("remote_server_uri")
        stage_staging=config.get("mlflow_config",{}).get("stage",{})
        stage_production= config.get("mlflow_config",{}).get("")
        # Set tracking URI - prioritize remote_server_uri if available
        active_uri = remote_server_uri if remote_server_uri else tracking_uri
        if active_uri:
            logger.info(f"Setting MLflow tracking URI to: {active_uri}")
            mlflow.set_tracking_uri(active_uri)
        client = MlflowClient()

        model_name = results.get("model_name")
        if not model_name:
            raise ValueError("No model name found in results")

        # Get X_test and y_test from results
        X_test = results.get("X_test")
        y_test = results.get("y_test")

        if X_test is None or y_test is None:
            raise ValueError("Test data not available for evaluation")

        # Load models
        stage_model, prod_model = load_models(client, model_name,stage_staging, stage_production)

        # Evaluate staging model
        stage_metrics = evaluate_model(config, stage_model, X_test, y_test)
        results["metrics"] = stage_metrics

        # Get staging model version
        staging_versions = client.get_latest_versions(model_name, stages=[stage_staging])
        if not staging_versions:
            raise ValueError(f"No Staging version found for model: {model_name}")

        version = staging_versions[0].version

        # If production model exists, compare metrics
        promote_model = True
        if prod_model:
            prod_metrics = evaluate_model(config, prod_model, X_test, y_test)
            is_better, diff_metrics = compare_metrics(config, stage_metrics, prod_metrics)
            results["diff_metrics"] = diff_metrics

            # Only promote if staging model is better
            promote_model = is_better
            logger.info(
                f"Model comparison: is_better={is_better}, diff_metrics={diff_metrics}"
            )

        if promote_model:
            logger.info(f"Promoting model {model_name} version {version} to Production")
            # Transition to production
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage_production,
                archive_existing_versions=True,  # Archive existing production model
            )

            # Add promotion metadata as tags
            client.set_model_version_tag(
                name=model_name, version=version, key="promotion_strategy", value="direct"
            )

            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_at",
                value=str(int(time.time())),
            )

            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_by",
                value=config.get("author", "automated_pipeline"),
            )

            if results and "metrics" in results:
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="evaluation_metrics",
                    value=json.dumps(results["metrics"]),
                )

            logger.info(
                f"Model {model_name} version {version} successfully promoted to Production"
            )
            return {
                    "status": "success",
                    "model_name": model_name,
                    "version": version,
                    "stage": stage_production,
                    "is_promoted": True,
                    "metrics": stage_metrics,
                    "diff_metrics": diff_metrics
                }
        else:
            logger.info(
                f"Model {model_name} version {version} NOT promoted: performance not better than current production"
            )
            
            return {
                "status": "success",
                "model_name": model_name,
                "version": version,
                "stage": stage_staging,  # Remains in staging
                "is_promoted": False,
                "metrics": stage_metrics,
                "diff_metrics": diff_metrics
            }
        
    except Exception as e:
        logger.error(f"Error in promote_to_production: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }
