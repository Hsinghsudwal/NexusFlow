import pandas as pd
import os
import mlflow
import time
import tempfile
import json
from urllib.parse import urlparse
from datetime import datetime
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from src.utils.config_manager import ConfigManager
from utils.logger import logger


# class MLflowManager:
def model_tracking(config_file, results):
    """Initialize MLflow manager with configuration"""
    try:
        config = ConfigManager.load_file(config_file)

        # Get MLflow configuration
        problem_type = config.get("base", {}).get("problem_type", "regression")
        tracking_uri = config.get("mlflow_config", {}).get("mlflow_tracking_uri")
        experiment_name = config.get("mlflow_config", {}).get("experiment_name")
        description = config.get("mlflow_config", {}).get("description", "")
        remote_server_uri = config.get("mlflow_config", {}).get("remote_server_uri")
        auto_register = config.get("mlflow_config", {}).get("auto_register", False)

        logger.info("Starting mlflow tracking")

        # Set tracking URI - prioritize remote_server_uri if available
        active_uri = remote_server_uri if remote_server_uri else tracking_uri
        if active_uri:
            logger.info(f"Setting MLflow tracking URI to: {active_uri}")
            mlflow.set_tracking_uri(active_uri)
        # else:
        #     logger.warning("No MLflow tracking URI specified, using default")

        # Get URI scheme to determine if it's a local file or remote server
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
        is_remote = tracking_url_type != "file"

        # Set tracking URI
        # if tracking_uri:
        #     mlflow.set_tracking_uri(tracking_uri)

        # Create MLflow client
        client = MlflowClient()

        # Check if experiment exists, create if not
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            experiment_id = client.create_experiment(
                experiment_name, tags={"description": description}
            )
        else:
            experiment_id = exp.experiment_id

        # Extract necessary data from results
        X_train = results.get("X_train")
        X_test = results.get("X_test")
        y_test = results.get("y_test")
        best_model = results.get("best_model")
        best_model_name = results.get("best_model_name", "unknown_model")
        metrics = results.get("eval_metrics", {})
        preprocessor = results.get("preprocessor")

        if best_model is None:
            return {"error": "No model found in results"}

        # Start MLflow run
        project_name = config.get("project_name", "unknown_project")
        model_name = f"{best_model_name}_{project_name}"
        run_name = f"{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            run_id = run.info.run_id

            # Log tags
            mlflow.set_tag("Project", project_name)
            mlflow.set_tag("Dev", config.get("author", "unknown_author"))
            mlflow.set_tag("ModelType", best_model_name)

            # Log basic parameters
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param(
                "target_column", config.get("base", {}).get("target_column", {})
            )
            mlflow.log_param("evaluation_metric", str(metrics))
            mlflow.log_param("cv_folds", config.get("base", {}).get("cv_folds", 3))
            mlflow.log_param(
                "train_test_split", config.get("base", {}).get("test_size", 0.2)
            )
            mlflow.log_param("random_state", config.get("base", {}).get("random_state", 42))
            mlflow.log_param("selected_model", best_model_name)

            # Log dataset shapes
            if X_train is not None:
                mlflow.log_param("X_train_shape", str(X_train.shape))
            if X_test is not None:
                mlflow.log_param("X_test_shape", str(X_test.shape))

            # Log preprocessor details if available
            if preprocessor is not None:
                if hasattr(preprocessor, "preprocessor") and preprocessor.preprocessor:
                    transformers = preprocessor.preprocessor.transformers_
                    for name, transformer, columns in transformers:
                        mlflow.log_param(f"preprocessor_{name}_columns", str(columns))

                # Log preprocessor as artifact
                mlflow.sklearn.log_model(preprocessor, "preprocessor")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                # print(metric_name , metric_value)
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)


            # Log appropriate metrics based on problem type
            if problem_type == "classification":
                # Classification metrics
                for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    if metric_name in metrics:
                        mlflow.log_metric(metric_name, metrics[metric_name])
            else:
                # Regression metrics
                for metric_name in ["rmse", "mae", "r2", "explained_variance"]:
                    if metric_name in metrics:
                        mlflow.log_metric(metric_name, metrics[metric_name])

            # Log evaluation report as JSON
            evaluation_report = results.get("eval_metrics", {})
            with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
                json.dump(evaluation_report, f)
                report_path = f.name
            mlflow.log_artifact(report_path, "evaluation_report")
            os.remove(report_path)

            # Log evaluation images
            # img=results.get("eval_metadata",{}).get("plots",{})
            artifact_manager = results["artifact_manager"]
            plots = results.get("eval_metadata", {}).get(
                "plots", {}
            )  # evaluation_report.get("eval_metadata", {}).get("plots", {})
            for plot_name, plot_path in plots.items():
                plot_full_path = artifact_manager.resolve_path(plot_path)
                if os.path.exists(plot_full_path) and plot_full_path.endswith(
                    (".png", ".jpg", ".jpeg", ".svg")
                ):
                    mlflow.log_artifact(plot_full_path, "evaluation_plots")

            # Create model signature
            signature = None
            if X_test is not None and best_model is not None:
                if y_test is not None:
                    # Use ground truth for better signature
                    signature = infer_signature(X_test, y_test)
                else:
                    # Use predictions
                    signature = infer_signature(X_test, best_model.predict(X_test))

                log_args = {
                    "sk_model": best_model,
                    "artifact_path": "model",
                }

                if signature:
                    log_args["signature"] = signature

                # Only add registered_model_name if using remote tracking and auto_register is enabled
                if is_remote and auto_register:
                    log_args["registered_model_name"] = model_name

                # Log the model
                mlflow.sklearn.log_model(**log_args)

                # Log model metadata
                mlflow.log_param("final_model_name", model_name)
                mlflow.log_param("model_timestamp", datetime.now().isoformat())

                # For local tracking server but with explicit model registration requested
                model_version = None

                if (not is_remote or not auto_register) and config.get(
                    "mlflow_config", {}
                ).get("register_model", False):

                    model_uri = f"runs:/{run_id}/model"
                    result = mlflow.register_model(model_uri, model_name)
                    model_version = result.version

                    # Add description if provided
                    if description and result:
                        client.update_model_version(
                            name=model_name, version=result.version, description=description
                        )

                    # Add metrics as tags for easy filtering
                    if metrics and result:
                        for metric_name, metric_value in metrics.items():
                            client.set_model_version_tag(
                                name=model_name,
                                version=result.version,
                                key=f"metric.{metric_name}",
                                value=str(metric_value),
                            )

                        # Add timestamp for when model was registered
                        client.set_model_version_tag(
                            name=model_name,
                            version=result.version,
                            key="registered_at",
                            value=str(int(time.time())),
                        )

                elif is_remote and auto_register:
                    # Get the latest version if auto-registered

                    latest_model = client.get_latest_versions(model_name, stages=["None"])
                    if latest_model:
                        model_version = latest_model[0].version
                logger.info("mlflow model register completed")
                return {
                    "run_id": run_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "experiment_id": experiment_id,
                    "tracking_uri": mlflow.get_tracking_uri(),
                }
    except Exception as e:
        logger.error(f"Model tracking failed: {str(e)}")
        return{"error": f"Model tracking failed: {str(e)}"}
