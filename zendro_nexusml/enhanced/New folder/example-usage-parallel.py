"""
Example usage of our custom MLOps framework
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import our custom framework
from custom_mlops import (
    Pipeline, StackingPipeline, 
    DataLoaderStep, PreprocessorStep, ModelTrainerStep,
    PredictorStep, EvaluatorStep, StackingStep
)

# Custom data loader function
def load_diabetes_data(context):
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': diabetes.feature_names
    }

# Custom preprocessing function
def standardize_data(data, context):
    # Create a copy to avoid modifying the original
    processed = data.copy()
    
    # Scale features
    scaler = StandardScaler()
    processed['X_train'] = scaler.fit_transform(data['X_train'])
    processed['X_test'] = scaler.transform(data['X_test'])
    
    # Store the scaler in the processed data
    processed['scaler'] = scaler
    
    return processed

# Example 1: Simple sequential pipeline
def run_simple_pipeline(parallel=False):
    print(f"\n--- Running Simple Pipeline (Parallel: {parallel}) ---")
    
    # Create pipeline
    pipeline = Pipeline(name="DiabetesPrediction", storage_path="./ml_artifacts")
    
    # Create steps
    data_loader = DataLoaderStep(name="DiabetesDataLoader", data_loader_fn=load_diabetes_data)
    preprocessor = PreprocessorStep(name="StandardScaler", preprocessor_fn=standardize_data)
    model_trainer = ModelTrainerStep(
        name="RidgeRegressor", 
        model_class=Ridge, 
        model_params={'alpha': 1.0}
    )
    predictor = PredictorStep(name="RidgePredictor")
    evaluator = EvaluatorStep(
        name="RegressionEvaluator",
        metrics={
            'mse': mean_squared_error,
            'r2': r2_score
        }
    )
    
    # Connect steps
    preprocessor.connect(data_loader)
    model_trainer.connect(preprocessor)
    predictor.connect(model_trainer)
    predictor.connect(data_loader)  # Also need data for prediction
    evaluator.connect(predictor)
    evaluator.connect(data_loader)  # Need true values for evaluation
    
    # Add steps to pipeline
    pipeline.add_step(data_loader)
    pipeline.add_step(preprocessor)
    pipeline.add_step(model_trainer)
    pipeline.add_step(predictor)
    pipeline.add_step(evaluator)
    
    # Run pipeline
    artifacts = pipeline.run(parallel=parallel)
    
    # Print results
    metrics = artifacts['RegressionEvaluator'].data
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    return pipeline

# Example 2: Stacking pipeline with multiple models
def run_stacking_pipeline(parallel=False):
    print(f"\n--- Running Stacking Pipeline (Parallel: {parallel}) ---")
    
    # Create stacking pipeline
    pipeline = StackingPipeline(name="DiabetesStackingPipeline", storage_path="./ml_artifacts")
    
    # Create common steps
    data_loader = DataLoaderStep(name="DiabetesDataLoader", data_loader_fn=load_diabetes_data)
    preprocessor = PreprocessorStep(name="StandardScaler", preprocessor_fn=standardize_data)
    
    # Add data loading and preprocessing to pipeline
    pipeline.add_step(data_loader)
    pipeline.add_step(preprocessor)
    preprocessor.connect(data_loader)
    
    # Create base models
    ridge_model = ModelTrainerStep(
        name="RidgeModel", 
        model_class=Ridge, 
        model_params={'alpha': 1.0}
    )
    
    lasso_model = ModelTrainerStep(
        name="LassoModel", 
        model_class=Lasso, 
        model_params={'alpha': 0.1}
    )
    
    rf_model = ModelTrainerStep(
        name="RandomForestModel", 
        model_class=RandomForestRegressor, 
        model_params={'n_estimators': 100, 'max_depth': 5}
    )
    
    # Connect base models to preprocessor
    ridge_model.connect(preprocessor)
    lasso_model.connect(preprocessor)
    rf_model.connect(preprocessor)
    
    # Create predictors for base models
    ridge_predictor = PredictorStep(name="RidgePredictor")
    lasso_predictor = PredictorStep(name="LassoPredictor")
    rf_predictor = PredictorStep(name="RandomForestPredictor")
    
    # Connect predictors to models and data
    ridge_predictor.connect(ridge_model)
    ridge_predictor.connect(preprocessor)
    
    lasso_predictor.connect(lasso_model)
    lasso_predictor.connect(preprocessor)
    
    rf_predictor.connect(rf_model)
    rf_predictor.connect(preprocessor)
    
    # Create stacking step
    stacker = StackingStep(
        name="StackingModel",
        meta_model_class=Ridge,
        meta_model_params={'alpha': 0.5}
    )
    
    # Connect stacker to base model predictors
    stacker.connect(ridge_predictor)
    stacker.connect(lasso_predictor)
    stacker.connect(rf_predictor)
    stacker.connect(data_loader)  # Need target values for meta-model training
    
    # Create predictor for stacked model
    stacked_predictor = PredictorStep(
        name="StackedPredictor",
        predict_fn=lambda model, data, context: model['meta_model'].predict(
            np.column_stack([
                m.predict(data['X_test']) for m in model['base_models']
            ])
        )
    )
    
    # Connect stacked predictor
    stacked_predictor.connect(stacker)
    