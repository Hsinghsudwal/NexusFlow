import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath

# Define component for data preparation
def data_preparation(
    config_path: str,
    output_train_path: OutputPath(str),
    output_test_path: OutputPath(str)
):
    """Prepares data for model training"""
    import yaml
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import pickle
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Connect to database
    import psycopg2
    conn = psycopg2.connect(
        host=config['db_host'],
        database=config['db_name'],
        user=config['db_user'],
        password=config['db_password']
    )
    
    # Query data
    query = """
    SELECT 
        customer_id, 
        tenure, 
        contract, 
        monthly_charges, 
        total_charges,
        internet_service, 
        online_security,
        tech_support, 
        streaming_tv, 
        streaming_movies,
        payment_method,
        churn
    FROM customers
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Preprocess data
    # One-hot encode categorical variables
    cat_cols = ['contract', 'internet_service', 'online_security', 
                'tech_support', 'streaming_tv', 'streaming_movies', 'payment_method']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Split features and target
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save data splits
    with open(output_train_path, 'wb') as f:
        pickle.dump((X_train, y_train), f)
        
    with open(output_test_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)

# Define component for model training
def model_training(
    training_data_path: InputPath(str),
    model_config_path: str,
    model_output_path: OutputPath(str),
):
    """Trains a machine learning model on the prepared data"""
    import pickle
    import yaml
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    
    # Load config
    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    with open(training_data_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # Configure MLflow
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    
    # Start an MLflow run
    with mlflow.start_run(run_name=config['run_name']) as run:
        run_id = run.info.run_id
        
        # Log parameters
        mlflow.log_params(config['model_params'])
        
        # Train model based on algorithm
        if config['algorithm'] == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=config['model_params']['n_estimators'],
                max_depth=config['model_params']['max_depth'],
                random_state=42
            )
        elif config['algorithm'] == 'logistic_regression':
            model = LogisticRegression(
                C=config['model_params']['C'],
                solver=config['model_params']['solver'],
                random_state=42
            )
        elif config['algorithm'] == 'xgboost':
            model = XGBClassifier(
                n_estimators=config['model_params']['n_estimators'],
                max_depth=config['model_params']['max_depth'],
                learning_rate=config['model_params']['learning_rate'],
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {config['algorithm']}")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model for next steps in pipeline
        with open(model_output_path, 'wb') as f:
            pickle.dump((model, run_id), f)

# Define component for model evaluation
def model_evaluation(
    model_path: InputPath(str),
    test_data_path: InputPath(str),
    evaluation_config_path: str,
    metrics_output_path: OutputPath(str),
):
    """Evaluates the trained model and logs metrics"""
    import pickle
    import yaml
    import json
    import numpy as np
    import mlflow
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Load config
    with open(evaluation_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and run_id
    with open(model_path, 'rb') as f:
        model, run_id = pickle.load(f)
    
    # Load test data
    with open(test_data_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # Configure MLflow
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
    }
    
    # Log metrics in MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    
    # Save metrics for deployment decision
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)

# Define component for model deployment
def model_deployment(
    model_path: InputPath(str),
    metrics_path: InputPath(str),
    deployment_config_path: str,
    deployment_output_path: OutputPath(str),
):
    """Deploys the model if metrics meet threshold requirements"""
    import pickle
    import yaml
    import json
    import mlflow
    import mlflow.sklearn
    from datetime import datetime
    
    # Load config
    with open(deployment_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load model and run_id
    with open(model_path, 'rb') as f:
        model, run_id = pickle.load(f)
    
    # Configure MLflow
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    
    # Check if model meets deployment criteria
    deploy = all(
        metrics[metric_name] >= threshold 
        for metric_name, threshold in config['metric_thresholds'].items()
    )
    
    result = {
        'deployed': deploy,
        'model_uri': None,
        'model_version': None,
        'deployment_timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    if deploy:
        # Register model in MLflow Model Registry
        with mlflow.start_run(run_id=run_id):
            registered_model = mlflow.register_model(
                f"runs:/{run_id}/model",
                config['model_name']
            )
            
            # Transition to Production if set in config
            if config.get('auto_transition_to_production', False):
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=config['model_name'],
                    version=registered_model.version,
                    stage="Production"
                )
            
            result['model_uri'] = f"models:/{config['model_name']}/{registered_model.version}"
            result['model_version'] = registered_model.version
    
    # Save deployment result
    with open(deployment_output_path, 'w') as f:
        json.dump(result, f)

# Create pipeline
@dsl.pipeline(
    name='Customer Churn Training Pipeline',
    description='End-to-end pipeline for training and deploying customer churn prediction model'
)
def churn_pipeline(
    data_config_path: str = '/config/data.yaml',
    model_config_path: str = '/config/model.yaml',
    evaluation_config_path: str = '/config/evaluation.yaml',
    deployment_config_path: str = '/config/deployment.yaml'
):
    # Convert python functions to pipeline components
    data_prep_op = func_to_container_op(data_preparation, base_image='python:3.9')
    training_op = func_to_container_op(model_training, base_image='python:3.9')
    evaluation_op = func_to_container_op(model_evaluation, base_image='python:3.9')
    deployment_op = func_to_container_op(model_deployment, base_image='python:3.9')
    
    # Define pipeline
    data_prep_task = data_prep_op(data_config_path)
    
    training_task = training_op(
        data_prep_task.outputs['output_train_path'],
        model_config_path
    )
    
    evaluation_task = evaluation_op(
        training_task.outputs['model_output_path'],
        data_prep_task.outputs['output_test_path'],
        evaluation_config_path
    )
    
    deployment_task = deployment_op(
        training_task.outputs['model_output_path'],
        evaluation_task.outputs['metrics_output_path'],
        deployment_config_path
    )
    
    # Add resource requirements
    data_prep_task.set_cpu_request('1').set_memory_request('2G')
    training_task.set_cpu_request('2').set_memory_request('4G').set_gpu_limit('1')
    evaluation_task.set_cpu_request('1').set_memory_request('2G')
    deployment_task.set_cpu_request('1').set_memory_request('1G')
    
    # Add retries
    data_prep_task.set_retry(3)
    training_task.set_retry(2)
    evaluation_task.set_retry(2)
    deployment_task.set_retry(3)

# Compile pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path='churn_pipeline.yaml'
    )
