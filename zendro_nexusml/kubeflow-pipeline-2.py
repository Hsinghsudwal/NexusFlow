# pipelines/training_pipeline.py
import kfp
from kfp import dsl
from kfp.components import func_to_container_op
import os
import yaml
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define component functions
@func_to_container_op
def data_extraction_op(config_path: str) -> str:
    """
    Data extraction component for Kubeflow pipeline.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        str: Path to extracted data
    """
    import yaml
    import os
    import sys
    
    # Add project directory to path
    sys.path.append('/app')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import data ingestion
    from src.data.ingestion import DataIngestion
    
    # Initialize data ingestion
    data_ingestion = DataIngestion(config)
    
    # Extract data
    customer_data = data_ingestion.extract_customer_data()
    
    # Save data
    data_path = data_ingestion.save_to_processed(customer_data, 'churn_data_pipeline')
    
    return data_path

@func_to_container_op
def data_transformation_op(data_path: str, config_path: str) -> str:
    """
    Data transformation component for Kubeflow pipeline.
    
    Args:
        data_path (str): Path to input data
        config_path (str): Path to config file
        
    Returns:
        str: Path to transformed data
    """
    import yaml
    import os
    import sys
    import pandas as pd
    
    # Add project directory to path
    sys.path.append('/app')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import data transformation
    from src.data.transformation import DataTransformation
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Initialize transformation
    data_transformation = DataTransformation(config)
    
    # Engineer features
    df_with_features = data_transformation.engineer_features(df)
    
    # Save transformed data
    transformed_data_path = os.path.join(config['processed_data_dir'], 'transformed_data.parquet')
    df_with_features.to_parquet(transformed_data_path, index=False)
    
    # Save preprocessing pipeline
    pipeline_path = data_transformation.save_pipeline()
    
    # Return paths as JSON
    result = {
        'transformed_data_path': transformed_data_path,
        'pipeline_path': pipeline_path
    }
    
    return json.dumps(result)

@func_to_container_op
def split_data_op(transformed_data_path: str, config_path: str) -> str:
    """
    Data splitting component for Kubeflow pipeline.
    
    Args:
        transformed_data_path (str): Path to transformed data
        config_path (str): Path to config file
        
    Returns:
        str: Paths to train and test datasets
    """
    import yaml
    import os
    import sys
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import json
    
    # Add project directory to path
    sys.path.append('/app')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse transformed_data_path if it's a JSON string
    if transformed_data_path.startswith('{'):
        paths = json.loads(transformed_data_path)
        transformed_data_path = paths['transformed_data_path']
    
    # Load data
    df = pd.read_parquet(transformed_data_path)
    
    # Get features and target
    X = df.drop(columns=[config['target_column']])
    y = df[config['target_column']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=y
    )
    
    # Create data directory
    split_data_dir = os.path.join(config['processed_data_dir'], 'split')
    os.makedirs(split_data_dir, exist_ok=True)
    
    # Save split datasets
    train_features_path = os.path.join(split_data_dir, 'X_train.parquet')
    train_target_path = os.path.join(split_data_dir, 'y_train.parquet')
    test_features_path = os.path.join(split_data_dir, 'X_test.parquet')
    test_target_path = os.path.join(split_data_dir, 'y_test.parquet')
    
    X_train.to_parquet(train_features_path, index=False)
    pd.DataFrame(y_train).to_parquet(train_target_path, index=False)
    X_test.to_parquet(test_features_path, index=False)
    pd.DataFrame(y_test).to_parquet(test_target_path, index=False)
    
    # Return paths as JSON
    result = {
        'train_features_path': train_features_path,
        'train_target_path': train_target_path,
        'test_features_path': test_features_path,
        'test_target_path': test_target_path
    }
    
    return json.dumps(result)

@func_to_container_op
def train_model_op(split_data_paths: str, config_path: str) -> str:
    """
    Model training component for Kubeflow pipeline.
    
    Args:
        split_data_paths (str): Paths to split datasets (JSON)
        config_path (str): Path to config file
        
    Returns:
        str: Path to trained model
    """
    import yaml
    import os
    import sys
    import pandas as pd
    import json
    
    # Add project directory to path
    sys.path.append('/app')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import model trainer
    from src.models.train import ModelTrainer
    
    # Parse data paths
    data_paths = json.loads(split_data_paths)
    
    # Load data
    X_train = pd.read_parquet(data_paths['train_features_path'])
    y_train = pd.read_parquet(data_paths['train_target_path'])[config['target_column']]
    
    # Initialize trainer
    model_trainer = ModelTrainer(config)
    
    # Train model
    model = model_trainer.train(X_train, y_train, model_type=config['model_type'])
    
    # Save model
    model_path = model_trainer.save_model()
    
    return model_path

@func_to_container_op
def evaluate_model_op(model_path: str, split_data_paths: str, config_path: str) -> str:
    """
    Model evaluation component for Kubeflow pipeline.
    
    Args:
        model_path (str): Path to trained model
        split_data_paths (str): Paths to split datasets (JSON)
        config_path (str): Path to config file
        
    Returns:
        str: Evaluation metrics (JSON)
    """
    import yaml
    import os
    import sys
    import pandas as pd
    import json
    
    # Add project directory to path
    sys.path.append('/app')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import model trainer
    from src.models.train import ModelTrainer
    
    # Parse data paths
    data_paths = json.loads(split_data_paths)
    
    # Load data
    X_test = pd.read_parquet(data_paths['test_features_path'])
    y_test = pd.read_parquet(data_paths['test_target_path'])[config['target_column']]
    
    # Initialize trainer
    model_trainer = ModelTrainer(config)
    
    # Load model
    model_trainer.load_model(model_path)
    
    # Evaluate model
    metrics = model_trainer.evaluate(X_test, y_test)
    
    # Save metrics
    metrics_path = os.path.join(config['model_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    return json.dumps(metrics)

@func_to_container_op
def register_model_op(model_path: str, metrics: str, config_path: str, pipeline_path: str) -> str:
    """
    Model registration component for Kubeflow pipeline.
    
    Args:
        model_path (str): Path to trained model
        metrics (str): Evaluation metrics (JSON)
        config_path (str): Path to config file
        pipeline_path (str): Path to preprocessing pipeline
        
    Returns:
        str: Path to registered model
    """
    import yaml
    import os
    import sys
    import json
    import mlflow
    import mlflow.sklearn
    from datetime import datetime
    import shutil
    
    # Add project directory to path
    sys.path.append('/app')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up MLflow
    mlflow_tracking_uri = config.get('mlflow_tracking_uri')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = config.get('mlflow_experiment', 'customer_churn')
    
    # Parse metrics if it's a string
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    
    # Check if model meets registration criteria
    registration_threshold = config.get('registration_threshold', 0.7)
    metric_to_check = config.get('registration_metric', 'f1')
    
    if metrics[metric_to_check] >= registration_threshold:
        # Model meets criteria, register it
        with mlflow.start_run(experiment_name=experiment_name, run_name=f"model_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log model details
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("pipeline_path", pipeline_path)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(pipeline_path)
            
            # Create production model directory
            prod_model_dir = os.path.join(config['model_dir'], 'production')
            os.makedirs(prod_model_dir, exist_ok=True)
            
            # Copy model and pipeline to production directory
            prod_model_path = os.path.join(prod_model_dir, 'model.joblib')
            prod_pipeline_path = os.path.join(prod_model_dir, 'pipeline.joblib')
            
            shutil.copy(model_path, prod_model_path)
            shutil.copy(pipeline_path, prod_pipeline_path)
            
            # Create model info file
            model_info = {
                'model_path': prod_model_path,
                'pipeline_path': prod_pipeline_path,
                'metrics': metrics,
                'registered_at': datetime.now().isoformat(),
                'model_type': config['model_type'],
                'mlflow_run_id': mlflow.active_run().info.run_id
            }
            
            model_info_path = os.path.join(prod_model_dir, 'model_info.json')
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return json.dumps(model_info)
    else:
        # Model doesn't meet criteria
        return json.dumps({
            'status': 'rejected',
            'reason': f"{metric_to_check} score {metrics[metric_to_check]} below threshold {registration_threshold}"
        })

# Define the pipeline
@dsl.pipeline(
    name='Customer Churn Training Pipeline',
    description='End-to-end training pipeline for customer churn prediction'
)
def churn_training_pipeline(config_path: str):
    """
    Kubeflow pipeline for customer churn prediction model training.
    
    Args:
        config_path (str): Path to configuration file
    """
    # Extract data
    extract_data_task = data_extraction_op(config_path)
    
    # Transform data
    transform_data_task = data_transformation_op(extract_data_task.output, config_path)
    
    # Split data
    split_data_task = split_data_op(transform_data_task.output, config_path)
    
    # Train model
    train_model_task = train_model_op(split_data_task.output, config_path)
    
    # Evaluate model
    evaluate_model_task = evaluate_model_op(
        train_model_task.output, 
        split_data_task.output, 
        config_path
    )
    
    # Register model if it meets criteria
    register_model_task = register_model_op(
        train_model_task.output,
        evaluate_model_task.output,
        config_path,
        transform_data_task.output
    )

# Compile the pipeline
def compile_pipeline(output_path='customer_churn_pipeline.yaml'):
    """
    Compile the Kubeflow pipeline.
    
    Args:
        output_path (str): Path to save compiled pipeline
    """
    kfp.compiler.Compiler().compile(
        pipeline_func=churn_training_pipeline,
        package_path=output_path
    )

if __name__ == '__main__':
    # Compile the pipeline
    compile_pipeline()
    
    logger.info("Pipeline compiled successfully")
