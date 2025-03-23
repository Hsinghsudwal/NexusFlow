import logging
import pandas as pd
import numpy as np
from prefect import task, get_run_logger

logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """
    Pipeline for preparing data before feature engineering
    Handles data cleaning, type conversions, and basic transformations
    """
    
    def __init__(self, config):
        """Initialize the data preparation pipeline with configuration"""
        self.config = config
        self.target_column = config["data"]["target_column"]
        self.id_column = config["data"]["id_column"]
        self.drop_features = config["features"].get("drop_features", [])
    
    def run(self, data):
        """Run the data preparation pipeline"""
        logger.info("Starting data preparation")
        
        # Make a copy of the data to avoid modifying the original
        processed_data = data.copy()
        
        # Initialize preparation info
        prep_info = {
            "original_shape": data.shape,
            "transformed_columns": [],
            "dropped_columns": [],
            "parameters": {}
        }
        
        # Handle missing values (should be done in validation, but double-check)
        for column in processed_data.columns:
            if processed_data[column].isnull().sum() > 0:
                if column in self.config["features"]["numerical_features"]:
                    strategy = self.config["features"]["numerical_imputation_strategy"]
                    if strategy == "mean":
                        fill_value = processed_data[column].mean()
                    elif strategy == "median":
                        fill_value = processed_data[column].median()
                    else:  # zero
                        fill_value = 0
                    
                    processed_data[column].fillna(fill_value, inplace=True)
                    prep_info["transformed_columns"].append(f"{column} (imputed missing)")
                    
                elif column in self.config["features"]["categorical_features"]:
                    strategy = self.config["features"]["categorical_imputation_strategy"]
                    if strategy == "most_frequent":
                        fill_value = processed_data[column].mode()[0]
                    else:  # unknown
                        fill_value = "Unknown"
                        
                    processed_data[column].fillna(fill_value, inplace=True)
                    prep_info["transformed_columns"].append(f"{column} (imputed missing)")
        
        # Handle categorical variables with high cardinality
        for column in self.config["features"]["categorical_features"]:
            if column in processed_data.columns:
                value_counts = processed_data[column].value_counts()
                
                # If there are too many rare categories, group them
                if len(value_counts) > 10:  # Arbitrary threshold
                    # Find values that appear less than 1% of the time
                    rare_values = value_counts[value_counts / len(processed_data) < 0.01].index.tolist()
                    
                    if rare_values:
                        processed_data[column] = processed_data[column].apply(
                            lambda x: "Other" if x in rare_values else x
                        )
                        prep_info["transformed_columns"].append(f"{column} (grouped rare values)")
        
        # Handle potential data type issues
        for column in self.config["features"]["numerical_features"]:
            if column in processed_data.columns and not pd.api.types.is_numeric_dtype(processed_data[column]):
                try:
                    processed_data[column] = pd.to_numeric(processed_data[column], errors='coerce')
                    processed_data[column].fillna(processed_data[column].median(), inplace=True)
                    prep_info["transformed_columns"].append(f"{column} (converted to numeric)")
                except Exception as e:
                    logger.error(f"Error converting {column} to numeric: {str(e)}")
        
        # Drop specified features
        if self.drop_features:
            drop_cols = [c for c in self.drop_features if c in processed_data.columns]
            if drop_cols:
                processed_data = processed_data.drop(columns=drop_cols)
                prep_info["dropped_columns"].extend(drop_cols)
        
        # Feature-specific transformations for churn prediction
        
        # 1. Convert total_charges to numeric if it exists (in case it's a string)
        if 'total_charges' in processed_data.columns:
            if not pd.api.types.is_numeric_dtype(processed_data['total_charges']):
                processed_data['total_charges'] = pd.to_numeric(processed_data['total_charges'], errors='coerce')
                processed_data['total_charges'].fillna(0, inplace=True)
                prep_info["transformed_columns"].append("total_charges (converted to numeric)")
        
        # 2. Create derived features that might be useful
        if 'tenure' in processed_data.columns and 'monthly_charges' in processed_data.columns:
            # ARPU (Average Revenue Per User)
            if 'total_charges' in processed_data.columns:
                processed_data['arpu'] = processed_data['total_charges'] / processed_data['tenure'].replace(0, 1)
                prep_info["transformed_columns"].append("arpu (new derived feature)")
            
            # Check if total charges makes sense relative to monthly charges and tenure
            if 'total_charges' in processed_data.columns:
                expected_total = processed_data['monthly_charges'] * processed_data['tenure']
                # If total_charges is significantly off from expected, it might be an error
                large_discrepancy = abs(processed_data['total_charges'] - expected_total) > (0.2 * expected_total)
                if large_discrepancy.any():
                    # Correct total charges where the discrepancy is large
                    processed_data.loc[large_discrepancy, 'total_charges'] = processed_data.loc[large_discrepancy, 'monthly_charges'] * processed_data.loc[large_discrepancy, 'tenure']
                    prep_info["transformed_columns"].append("total_charges (corrected)")
        
        # Save preparation parameters
        prep_info["parameters"] = {
            "numerical_imputation_strategy": self.config["features"]["numerical_imputation_strategy"],
            "categorical_imputation_strategy": self.config["features"]["categorical_imputation_strategy"],
            "processed_shape": processed_data.shape
        }
        
        logger.info(f"Data preparation complete. Shape: {processed_data.shape}")
        return processed_data, prep_info
