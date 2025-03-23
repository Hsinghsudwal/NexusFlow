import functools
from datetime import datetime

# The custom @step decorator for pipeline components
def step(step_name):
    """
    A custom decorator for defining steps in the pipeline.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log start time and step details
            start_time = datetime.now()
            print(f"Starting step: {step_name} at {start_time}")
            
            # Execute the function (step)
            result = func(*args, **kwargs)
            
            # Log the end time and duration of the step
            end_time = datetime.now()
            print(f"Completed step: {step_name} at {end_time} | Duration: {end_time - start_time}")
            
            # Return the result of the function
            return result
        return wrapper
    return decorator

# The custom @pipeline decorator to define the entire pipeline
def pipeline(pipeline_name):
    """
    A custom decorator for defining a pipeline that organizes and executes steps.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            print(f"Starting pipeline: {pipeline_name} at {start_time}")
            
            # Execute the pipeline (which includes the steps)
            result = func(*args, **kwargs)
            
            end_time = datetime.now()
            print(f"Completed pipeline: {pipeline_name} at {end_time} | Duration: {end_time - start_time}")
            return result
        return wrapper
    return decorator

# Step 1: Define individual components using @step
@step("Data Preprocessing")
def preprocess_data(data):
    print("Preprocessing data...")
    # Some data preprocessing logic
    return {"X_train": "processed_data", "y_train": "processed_labels"}

@step("Model Training")
def train_model(data):
    print("Training model...")
    # Some model training logic
    model = "trained_model"
    return model

@step("Model Evaluation")
def evaluate_model(model, data):
    print("Evaluating model...")
    # Some evaluation logic
    accuracy = 0.95
    return accuracy

@step("Save Model")
def save_model(model, accuracy):
    print(f"Saving model with accuracy {accuracy}...")
    # Logic to save the model
    return model

@step("Model Deployment")
def deploy_model(model):
    print("Deploying model...")
    # Deployment logic
    return model

@step("Model Monitoring")
def monitor_model(model):
    print("Monitoring model...")
    # Monitoring logic
    return model

# Step 2: Define the pipeline using @pipeline
@pipeline("Full Model Pipeline")
def training_pipeline(raw_data):
    # Sequential execution of steps
    data = preprocess_data(raw_data)
    model = train_model(data)
    accuracy = evaluate_model(model, data)
    saved_model = save_model(model, accuracy)
    deployed_model = deploy_model(saved_model)
    monitor_model(deployed_model)
    
# Step 3: Run the pipeline
if __name__ == "__main__":
    raw_data = "sample_raw_data"  # This would be your actual input data
    training_pipeline(raw_data)
