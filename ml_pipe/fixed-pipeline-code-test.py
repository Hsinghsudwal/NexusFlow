import os
import json
import yaml
import pickle
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Config:
    def __init__(self, config_dict: Dict = None):
        self.config_dict = config_dict or {}

    def get(self, key: str, default: Any = None):
        return self.config_dict.get(key, default)

    @staticmethod
    def load_file(config_path: str):
        """Loads configuration from a YAML or JSON file."""
        try:
            with open(config_path, "r") as file:
                if config_path.endswith((".yml", ".yaml")):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
            return Config(config_data)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")


class ArtifactStore:
    """Stores and retrieves intermediate artifacts for the pipeline."""

    def __init__(self, config):
        self.config = config
        self.base_path = self.config.get("folder_path", {}).get(
            "artifacts", "artifacts"
        )
        os.makedirs(self.base_path, exist_ok=True)
        logging.info(f"Artifact store initialized at '{self.base_path}'")
        self.runs_metadata = {}  # Store run metadata

    def save_artifact(
        self,
        artifact: Any,
        subdir: str,
        name: str,
        run_id: str = None,
    ) -> None:
        """Save an artifact in the specified format."""
        if run_id:
            artifact_dir = os.path.join(self.base_path, run_id, subdir)
        else:
            artifact_dir = os.path.join(self.base_path, subdir)
            
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, name)

        if name.endswith(".pkl"):
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
        elif name.endswith(".csv"):
            if isinstance(artifact, pd.DataFrame):
                artifact.to_csv(artifact_path, index=False)
            else:
                raise ValueError("CSV format only supports pandas DataFrames.")
        elif name.endswith(".json"):
            with open(artifact_path, "w") as f:
                json.dump(artifact, f)
        else:
            raise ValueError(f"Unsupported format for {name}")
        
        logging.info(f"Artifact '{name}' saved to {artifact_path}")
        
        # Track artifact in run metadata
        if run_id:
            if run_id not in self.runs_metadata:
                self.runs_metadata[run_id] = {"artifacts": []}
            self.runs_metadata[run_id]["artifacts"].append(os.path.join(run_id, subdir, name))

    def load_artifact(
        self,
        subdir: str,
        name: str,
        run_id: str = None,
    ):
        """Load an artifact in the specified format."""
        if run_id:
            artifact_path = os.path.join(self.base_path, run_id, subdir, name)
        else:
            artifact_path = os.path.join(self.base_path, subdir, name)
            
        if os.path.exists(artifact_path):
            if name.endswith(".pkl"):
                with open(artifact_path, "rb") as f:
                    artifact = pickle.load(f)
            elif name.endswith(".csv"):
                artifact = pd.read_csv(artifact_path)
            elif name.endswith(".json"):
                with open(artifact_path, "r") as f:
                    artifact = json.load(f)
            else:
                raise ValueError(f"Unsupported format for {name}")
            logging.info(f"Artifact '{name}' loaded from {artifact_path}")
            return artifact
        else:
            logging.warning(f"Artifact '{name}' not found in {artifact_path}")
            return None

    def list_artifacts(self, run_id: str = None) -> List[str]:
        """List all artifacts for a specific run or in general."""
        if run_id and run_id in self.runs_metadata:
            return self.runs_metadata[run_id].get("artifacts", [])
        
        artifacts = []
        if run_id:
            base_dir = os.path.join(self.base_path, run_id)
        else:
            base_dir = self.base_path
            
        if not os.path.exists(base_dir):
            return artifacts
            
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), self.base_path)
                artifacts.append(rel_path)
        
        return artifacts


class Task:
    """Base class for all pipeline tasks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        
    def run(self, stack: 'StackPipeline', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task and return updated context."""
        logging.info(f"Starting task: {self.name}")
        start_time = pd.Timestamp.now()
        
        try:
            updated_context = self._run(stack, context)
            
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Task '{self.name}' completed in {duration:.2f} seconds")
            
            # Record task execution in the stack's run history
            task_result = {
                "name": self.name,
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            }
            
            if hasattr(stack, 'current_run_id') and stack.current_run_id:
                if stack.current_run_id not in stack.run_history:
                    stack.run_history[stack.current_run_id] = {"tasks": []}
                stack.run_history[stack.current_run_id]["tasks"].append(task_result)
            
            return updated_context
            
        except Exception as e:
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            logging.error(f"Task '{self.name}' failed after {duration:.2f} seconds: {str(e)}")
            
            # Record failure
            task_result = {
                "name": self.name,
                "status": "failed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": str(e)
            }
            
            if hasattr(stack, 'current_run_id') and stack.current_run_id:
                if stack.current_run_id not in stack.run_history:
                    stack.run_history[stack.current_run_id] = {"tasks": []}
                stack.run_history[stack.current_run_id]["tasks"].append(task_result)
            
            raise
    
    def _run(self, stack: 'StackPipeline', context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of task logic. Should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class StackPipeline:
    """Pipeline that manages the execution of a series of tasks."""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = Config(config)
        self.tasks = []
        self.run_history = {}
        self.current_run_id = None
        self.artifact_store = ArtifactStore(self.config)
        logging.info(f"Pipeline '{name}' initialized")
        
    def add_task(self, task: Task) -> None:
        """Add a task to the pipeline."""
        self.tasks.append(task)
        logging.info(f"Task '{task.name}' added to pipeline '{self.name}'")
        
    def run(self, context: Dict[str, Any] = None) -> str:
        """Run all tasks in the pipeline and return the run ID."""
        import uuid
        
        self.current_run_id = f"run_{uuid.uuid4().hex[:8]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = pd.Timestamp.now()
        
        # Initialize run in history
        self.run_history[self.current_run_id] = {
            "pipeline_name": self.name,
            "start_time": start_time.isoformat(),
            "status": "running",
            "tasks": []
        }
        
        logging.info(f"Starting pipeline '{self.name}' with run ID: {self.current_run_id}")
        
        # Initialize context
        current_context = context or {}
        
        try:
            # Execute each task in sequence
            for task in self.tasks:
                current_context = task.run(self, current_context)
                
            # Record successful completion
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            self.run_history[self.current_run_id].update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            })
            
            logging.info(f"Pipeline '{self.name}' completed in {duration:.2f} seconds")
            
            # Save run details as artifact
            self.artifact_store.save_artifact(
                self.run_history[self.current_run_id],
                subdir="runs",
                name=f"{self.current_run_id}_details.json",
                run_id=self.current_run_id
            )
            
            return self.current_run_id
            
        except Exception as e:
            # Record failure
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            self.run_history[self.current_run_id].update({
                "status": "failed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": str(e)
            })
            
            logging.error(f"Pipeline '{self.name}' failed after {duration:.2f} seconds: {str(e)}")
            
            # Save run details even for failed runs
            self.artifact_store.save_artifact(
                self.run_history[self.current_run_id],
                subdir="runs",
                name=f"{self.current_run_id}_details.json",
                run_id=self.current_run_id
            )
            
            raise
            
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get details about a specific pipeline run."""
        if run_id in self.run_history:
            return self.run_history[run_id]
        
        # Try to load from artifact store
        run_details = self.artifact_store.load_artifact(
            subdir="runs",
            name=f"{run_id}_details.json",
            run_id=run_id
        )
        
        if run_details:
            # Cache in memory
            self.run_history[run_id] = run_details
            return run_details
            
        logging.warning(f"No details found for run ID: {run_id}")
        return None


# ----------------------------- 
# Data Ingestion Task
# -----------------------------

class DataIngestionTask(Task):
    """Task for loading and splitting data."""

    def __init__(self, data_path: str):
        super().__init__(name="data_ingestion", description="Load and split data")
        self.data_path = data_path
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        # Load raw data
        df = pd.read_csv(self.data_path)
        logging.info(f"Loaded data from {self.data_path}, shape: {df.shape}")
        
        # Save the full dataset as an artifact
        stack.artifact_store.save_artifact(
            df, 
            subdir="raw", 
            name="full_dataset.csv",
            run_id=stack.current_run_id
        )

        # Split data
        test_size = stack.config.get("base.test_size", 0.2)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(f"Data split complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        # Save split datasets
        stack.artifact_store.save_artifact(
            train_data, 
            subdir="raw", 
            name="train_data.csv",
            run_id=stack.current_run_id
        )

        stack.artifact_store.save_artifact(
            test_data, 
            subdir="raw", 
            name="test_data.csv",
            run_id=stack.current_run_id
        )

        # Update context with data
        updated_context = {**context}
        updated_context["train_data"] = train_data
        updated_context["test_data"] = test_data
        
        return updated_context


# ----------------------------- 
# Data Processing Task
# -----------------------------

class DataProcessingTask(Task):
    """Task for preprocessing and feature engineering."""
    
    def __init__(self, preprocessing_steps: List[str] = None):
        super().__init__(name="data_processing", description="Preprocess data and engineer features")
        self.preprocessing_steps = preprocessing_steps or ["missing_values", "encoding", "scaling"]
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        train_data = context.get("train_data")
        test_data = context.get("test_data")
        
        if train_data is None or test_data is None:
            raise ValueError("Training or test data not found in context. Run data ingestion task first.")
            
        processed_train = self._process_data(train_data, is_train=True)
        processed_test = self._process_data(test_data, is_train=False)
        
        logging.info(f"Data processing complete. Train shape: {processed_train.shape}, Test shape: {processed_test.shape}")
        
        # Save processed datasets
        stack.artifact_store.save_artifact(
            processed_train,
            subdir="processed",
            name="processed_train_data.csv",
            run_id=stack.current_run_id
        )
        
        stack.artifact_store.save_artifact(
            processed_test,
            subdir="processed",
            name="processed_test_data.csv",
            run_id=stack.current_run_id
        )
        
        # Update context
        updated_context = {**context}
        updated_context["processed_train_data"] = processed_train
        updated_context["processed_test_data"] = processed_test
        
        return updated_context
        
    def _process_data(self, data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Apply preprocessing steps to the data."""
        processed_data = data.copy()
        
        # Here you would implement your actual preprocessing logic
        # This is just a placeholder for demonstration
        if "missing_values" in self.preprocessing_steps:
            # Handle missing values
            for col in processed_data.columns:
                if processed_data[col].dtype.kind in 'ifc':  # integer, float, complex
                    processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                else:
                    processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
                    
        if "encoding" in self.preprocessing_steps:
            # Simple categorical encoding (one-hot for demonstration)
            cat_columns = [col for col in processed_data.columns if processed_data[col].dtype == 'object']
            processed_data = pd.get_dummies(processed_data, columns=cat_columns, drop_first=True)
            
        if "scaling" in self.preprocessing_steps:
            # Simple scaling (just for demonstration)
            num_columns = [col for col in processed_data.columns if processed_data[col].dtype.kind in 'ifc']
            for col in num_columns:
                processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
                
        return processed_data


# ----------------------------- 
# Model Training Task
# -----------------------------

class ModelTrainingTask(Task):
    """Task for training a machine learning model."""
    
    def __init__(self, model_type: str = "default"):
        super().__init__(name="model_training", description="Train a machine learning model")
        self.model_type = model_type
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_train_data = context.get("processed_train_data")
        
        if processed_train_data is None:
            raise ValueError("Processed training data not found in context. Run data processing task first.")
            
        # Extract features and target
        target_col = stack.config.get("model.target_column", "target")
        feature_cols = [col for col in processed_train_data.columns if col != target_col]
        
        X_train = processed_train_data[feature_cols]
        y_train = processed_train_data[target_col]
        
        # Train model
        model = self._get_model()
        model.fit(X_train, y_train)
        logging.info(f"Model training completed for model type: {self.model_type}")
        
        # Save the model
        stack.artifact_store.save_artifact(
            model,
            subdir="models",
            name=f"{self.model_type}_model.pkl",
            run_id=stack.current_run_id
        )
        
        # Save feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            stack.artifact_store.save_artifact(
                importances,
                subdir="models",
                name="feature_importances.csv",
                run_id=stack.current_run_id
            )
            
        # Update context
        updated_context = {**context}
        updated_context["model"] = model
        updated_context["feature_cols"] = feature_cols
        
        return updated_context
        
    def _get_model(self):
        """Get the appropriate model based on model_type."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        else:  # default model
            return LogisticRegression(max_iter=1000, random_state=42)


# ----------------------------- 
# Model Evaluation Task
# -----------------------------

class ModelEvaluationTask(Task):
    """Task for evaluating model performance."""
    
    def __init__(self):
        super().__init__(name="model_evaluation", description="Evaluate model performance")
        
    def _run(self, stack: StackPipeline, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context.get("model")
        processed_test_data = context.get("processed_test_data")
        feature_cols = context.get("feature_cols")
        
        if model is None or processed_test_data is None or feature_cols is None:
            raise ValueError("Model, processed test data, or feature columns not found in context.")
            
        # Extract features and target
        target_col = stack.config.get("model.target_column", "target")
        X_test = processed_test_data[feature_cols]
        y_test = processed_test_data[target_col]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        logging.info(f"Model evaluation complete: {metrics}")
        
        # Save metrics
        stack.artifact_store.save_artifact(
            metrics,
            subdir="metrics",
            name="evaluation_metrics.json",
            run_id=stack.current_run_id
        )
        
        # Update context
        updated_context = {**context}
        updated_context["evaluation_metrics"] = metrics
        
        return updated_context
        
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted')),
                "recall": float(recall_score(y_true, y_pred, average='weighted')),
                "f1": float(f1_score(y_true, y_pred, average='weighted'))
            }
            
            # For binary classification, also calculate ROC AUC
            if len(set(y_true)) == 2:
                # If the model has predict_proba method
                if hasattr(self.model, 'predict_proba'):
                    y_prob = self.model.predict_proba(self.X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                    
            return metrics
        except Exception as e:
            logging.warning(f"Error calculating some metrics: {str(e)}")
            # Return basic metrics if advanced ones fail
            return {"accuracy": float(accuracy_score(y_true, y_pred))}


# ----------------------------- 
# Training Pipeline
# -----------------------------

class TrainingPipeline:
    """Main pipeline class that orchestrates the training workflow."""

    def __init__(self, data_path: str, config_path: str = "config/config.yml"):
        self.data_path = data_path
        self.config = Config.load_file(config_path).config_dict

        
    def run(self) -> Tuple[str, StackPipeline]:
        """Execute the training pipeline."""
        # Create stack
        stack = StackPipeline(name="training_pipeline", config=self.config)
        
        # Add tasks to pipeline
        stack.add_task(DataIngestionTask(self.data_path))
        stack.add_task(DataProcessingTask())
        stack.add_task(ModelTrainingTask(model_type=stack.config.get("model.type", "default")))
        stack.add_task(ModelEvaluationTask())
        
        # Run the pipeline
        try:
            run_id = stack.run()
            logging.info(f"Pipeline completed successfully with run ID: {run_id}")
            
            # Output run summary
            self._print_run_summary(stack, run_id)
            
            return run_id, stack
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise
            
    def _print_run_summary(self, stack: StackPipeline, run_id: str) -> None:
        """Print a summary of the pipeline run."""
        # List artifacts
        artifacts = stack.artifact_store.list_artifacts(run_id)
        print(f"\nRun ID: {run_id}")
        print("\nArtifacts:")
        for uri in artifacts:
            print(f"- {uri}")
            
        # Get run details
        run_details = stack.get_run_details(run_id)
        print("\nRun Details:")
        print(f"Pipeline: {run_details.get('pipeline_name')}")
        print(f"Status: {run_details.get('status')}")
        print(f"Duration: {run_details.get('duration_seconds', 0):.2f} seconds")
        
        # Check if run was successful
        if run_details.get("status") == "completed":
            print("Pipeline completed successfully")
            
            # Print evaluation metrics if available
            metrics = stack.artifact_store.load_artifact(
                "metrics", 
                "evaluation_metrics.json",
                run_id=run_id
            )
            if metrics:
                print("\nEvaluation Metrics:")
                for metric, value in metrics.items():
                    print(f"- {metric}: {value:.4f}")


# ----------------------------- 
# Example Usage
# ----------------------------- 

def main():
    """Main entry point for running the pipeline."""
    # Path to your data file
    data_path = "data.csv"
    
    # Create and run the pipeline
    try:
        pipeline = TrainingPipeline(data_path)
        run_id, stack = pipeline.run()
        
        print("\n" + "="*50)
        print("Pipeline execution complete!")
        print(f"Run ID: {run_id}")
        print("="*50)
        
        return run_id, stack
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        return None, None


if __name__ == "__main__":
    main()
