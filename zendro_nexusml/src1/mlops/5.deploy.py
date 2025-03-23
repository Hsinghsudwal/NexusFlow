
Let's add Model Deployment to the pipeline. This step involves:

Model Serialization: Save the trained model to disk (using formats like Pickle or SavedModel).
Deployment to Environment: Deploy the model into a production environment (e.g., LocalStack SageMaker, Kubernetes, Docker containers).
API Exposure: Expose the model as an API endpoint using frameworks like Flask or FastAPI for inference.
CI/CD Pipeline: Set up automation to handle the deployment and updates using CI/CD tools like GitHub Actions, Jenkins, or CircleCI.
Full Pipeline with Model Deployment:
Below is the updated pipeline with Model Deployment added. The following sections will cover:

Model Serialization
Model Deployment to LocalStack SageMaker / Docker / Kubernetes
API Exposure for Model Inference (Flask)
We'll also touch on setting up a basic CI/CD pipeline for automation (though this part can be expanded based on your specific CI/CD tools).

Updated Pipeline with Model Deployment:
python
Copy
import requests
import pandas as pd
import numpy as np
import boto3
import logging
import mlflow
import mlflow.sklearn
from prefect import task, Flow, context
from prefect.utilities.logging import get_logger
from botocore.exceptions import NoCredentialsError
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pickle
import os
from flask import Flask, request, jsonify
import docker
from prefect import task, Flow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger()

# Step 1: Data Collection from API with retries
@task(max_retries=3, retry_delay=retryable_delay)
def collect_data_from_api():
    url = "https://jsonplaceholder.typicode.com/posts"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError if the response code is not 200
        data = response.json()
        df = pd.DataFrame(data)
        logger.info("Data successfully collected from API.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Error while collecting data from API: {e}")
        raise  # Raise the exception to trigger a retry

# Step 2: Data Cleaning
@task
def clean_data(df):
    # Handle missing values
    df = df.dropna()  # Drop rows with missing values (can also use df.fillna() to fill missing values)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Filter outliers using Z-score (for numerical columns)
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]  # Removing rows with Z-score > 3 (common threshold for outliers)
    
    logger.info("Data cleaning completed: Missing values handled, duplicates removed, and outliers filtered.")
    return df_no_outliers

# Step 3: Feature Engineering
@task
def feature_engineering(df):
    # Example feature engineering: create a new column based on existing ones
    df['title_length'] = df['title'].apply(len)  # Example: Feature for length of the title text
    
    # Normalize/Standardize numerical columns (example: 'userId')
    scaler = StandardScaler()
    df['userId_normalized'] = scaler.fit_transform(df[['userId']])
    
    logger.info("Feature engineering completed: New features created and data standardized.")
    return df

# Step 4: Model Selection, Training, and Validation
@task
def train_and_validate_model(df):
    # Split data into training and validation sets
    X = df[['userId_normalized', 'title_length']]  # Features
    y = df['userId']  # Target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor()
    }
    
    # Log experiment with MLflow
    with mlflow.start_run():
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            logger.info(f"Model: {model_name}, RMSE: {rmse}")
            
            # Log model, hyperparameters, and metrics with MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, model_name)  # Log the model
    return models, X_train, y_train, X_val, y_val

# Step 5: Model Evaluation
@task
def evaluate_models(models, X_train, y_train, X_val, y_val):
    best_model = None
    best_model_name = None
    best_performance = -float('inf')  # To track the best model (based on RMSE or other metric)
    
    # Evaluate each model
    for model_name, model in models.items():
        y_pred = model.predict(X_val)
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        accuracy = accuracy_score(y_val, y_pred) if y_val.nunique() <= 2 else None  # For classification tasks
        precision = precision_score(y_val, y_pred, average='binary', zero_division=1) if y_val.nunique() <= 2 else None
        recall = recall_score(y_val, y_pred, average='binary', zero_division=1) if y_val.nunique() <= 2 else None
        f1 = f1_score(y_val, y_pred, average='binary', zero_division=1) if y_val.nunique() <= 2 else None
        
        logger.info(f"{model_name} Evaluation Metrics:")
        logger.info(f"RMSE: {rmse}")
        if accuracy is not None:
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")
        
        # Performance Monitoring: Check if the model meets predefined thresholds
        if rmse < best_performance or best_model is None:
            best_performance = rmse
            best_model = model
            best_model_name = model_name
    
    logger.info(f"Best performing model: {best_model_name} with RMSE: {best_performance}")
    return best_model, best_model_name, best_performance

# Step 6: Model Serialization (Pickle)
@task
def serialize_model(model, model_name="best_model"):
    model_filename = f"{model_name}.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    logger.info(f"Model serialized and saved to {model_filename}")
    return model_filename

# Step 7: Model Deployment to Docker (Example)
@task
def deploy_to_docker(model_filename):
    # Build a simple Docker image for serving the model using Flask
    docker_client = docker.from_env()
    dockerfile_content = """
    FROM python:3.8-slim
    RUN pip install flask scikit-learn
    COPY ./model.pkl /app/model.pkl
    COPY ./app.py /app/app.py
    WORKDIR /app
    CMD ["python", "app.py"]
    """
    
    # Write a simple Flask app to serve the model
    app_code = """
    from flask import Flask, request, jsonify
    import pickle
    import numpy as np
    
    app = Flask(__name__)
    
    # Load the serialized model
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5000)
    """
    
    # Save the app code
    with open("app.py", "w") as f:
        f.write(app_code)
    
    # Build and run the Docker container
    docker_client.images.build(path=".", tag="model-serving")
    container = docker_client.containers.run("model-serving", detach=True, ports={'5000/tcp': 5000})
    logger.info(f"Docker container running with model exposed on port 5000.")
    
    return container.id

# Step 8: Continuous Integration / Continuous Deployment (CI/CD)
@task
def setup_cicd_pipeline():
    # For simplicity, we simulate CI/CD integration, typically this would be done via GitHub Actions, Jenkins, or CircleCI.
    logger.info("Setting up CI/CD pipeline for deployment and model updates.")
    # In a real scenario, this would interact with your CI/CD platform for auto-deployment, testing, and versioning.
    return "CI/CD Pipeline setup completed."

# Full Pipeline with Prefect Flow
with Flow("MLops Data Pipeline with Model Deployment") as flow:
    # Task execution flow
    df = collect_data_from_api()
    df_cleaned = clean_data(df)
    df_featured = feature_engineering(df_cleaned)
    models, X_train, y_train, X_val, y_val = train_and_validate_model(df_featured)
    best_model, best_model_name, best_performance = evaluate_models(models, X_train, y_train, X_val, y_val)
    model_filename = serialize_model(best_model)
    container_id = deploy_to_docker(model_filename)
    cicd_status = setup_cicd_pipeline()

# Step 9: Monitor the Flow execution
def monitor_flow():
    try:
        # Execute the flow locally and monitor via the Prefect UI
        flow_state = flow.run()
        logger.info(f"Flow execution state: {flow_state}")
    except Exception as e:
        logger.error(f"Error while executing the flow: {e}")

if __name__ == "__main__":
    # Trigger the pipeline execution
    monitor_flow()
Model Deployment Breakdown:
Model Serialization:

The model is serialized using Pickle in the serialize_model task, and saved to a .pkl file. You can use other serialization formats such as SavedModel for TensorFlow models or joblib for scikit-learn models.
Model Deployment to Docker:

The model is served using a Flask API. The model is loaded from the .pkl file and predictions are made through a RESTful endpoint exposed at /predict.
We use Docker to containerize the model serving environment, ensuring that the model can be deployed consistently across different environments.
A Dockerfile is created dynamically, and the Docker image is built and run, exposing the model via a REST API on port 5000.
CI/CD Pipeline:

A task is added to simulate the integration of CI/CD tools like GitHub Actions or Jenkins. In a real-world application, this would involve automatically deploying the model updates after a successful code commit or test pass.
Exposing the Model API (Flask Example):
python
Copy
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the serialized model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
Running the Flask App in Docker:
Build and run the Docker container as mentioned in the deployment section. The model will be available for inference through the exposed /predict API endpoint.
Setting Up CI/CD:
To integrate CI/CD:

GitHub Actions: Create a .yml file for GitHub workflows to automate testing, building Docker images, and deploying the model.
Jenkins: Create Jenkins pipelines to build, test, and deploy your model.
CircleCI: Use CircleCI config to automate the workflow for model deployment.

