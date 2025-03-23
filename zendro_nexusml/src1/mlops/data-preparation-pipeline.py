import logging
import pandas as pd
import numpy as np
from prefect import task

logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """
    Pipeline for preparing data for feature engineering
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.target_column = config["data"]["target_column"]
        self.id_column = config["data"]["id_column"]
        self.drop_features = config["features"].get("drop_features", [])
        
    def run(self, data):
        """Run the data preparation pipeline"""
        logger.info("Starting data preparation")
        
        df = data.copy()
        
        # Track preparation steps and parameters
        prep_info = {
            "parameters": {},
            "steps_applied": []
        }
        
        # 1. Handle any remaining missing values
        df, missing_info = self._handle_missing_values(df)
        prep_info["steps_applied"].append("handle_missing_values")
        prep_info["parameters"]["missing_handling"] = missing_info
        
        # 2. Remove outliers if applicable (comment out if not desired)
        # df, outlier_info = self._handle_outliers(df)
        # prep_info["steps_applied"].append("handle_outliers")
        # prep_info["parameters"]["outlier_handling"] = outlier_info
        
        # 3. Drop unnecessary features
        df, dropped_info = self._drop_unnecessary_features(df)
        prep_info["steps_applied"].append("drop_unnecessary_features")
        prep_info["parameters"]["dropped_features"] = dropped_info
        
        # 4. Standardize column formats
        df, format_info = self._standardize_formats(df)
        prep_info["steps_applied"].append("standardize_formats")
        prep_info["parameters"]["format_standardization"] = format_info
        
        # 5. Target encoding
        df, target_info = self._encode_target(df)
        prep_info["steps_applied"].append("encode_target")
        prep_info["parameters"]["target_encoding"] = target_info
        
        logger.info(f"Data preparation complete: {len(df)} rows remaining")
        
        return df, prep_info
    
    def _handle_missing_values(self, df):
        """Handle any remaining missing values"""
        initial_missing = df.isnull().sum().sum()
        missing_info = {"initial_missing": int(initial_missing)}
        
        if initial_missing > 0:
            # Fill numeric columns with median
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if col != self.id_column and df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # Fill categorical columns with mode
            for col in df.select_dtypes(include=['object']).columns:
                if col != self.id_column and df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        final_missing = df.isnull().sum().sum()
        missing_info["final_missing"] = int(final_missing)
        missing_info["method"] = "median/mode"
        
        return df, missing_info
    
    def _handle_outliers(self, df):
        """Handle outliers in numerical columns"""
        outlier_info = {}
        
        # Only process numeric columns except ID and target
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != self.id_column and col != self.target_column]
        
        for col in numeric_cols:
            # Use IQR method to identify outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Cap the outliers instead of removing rows
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
                outlier_info[col] = {
                    "count": int(outlier_count),
                    "percentage": float(outlier_count / len(df) * 100),
                    "method": "capping",
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
                
                logger.info(f"Capped {outlier_count} outliers in column {col}")
        
        return df, outlier_info
    
    def _drop_unnecessary_features(self, df):
        """Drop unnecessary features specified in config"""
        dropped_info = {"features": []}
        
        # Drop features from config
        if self.drop_features:
            for feature in self.drop_features:
                if feature in df.columns:
                    df = df.drop(columns=[feature])
                    dropped_info["features"].append(feature)
                    logger.info(f"Dropped feature: {feature}")
        
        # Check for constant features (all values are the same)
        constant_features = []
        for col in df.columns:
            if col != self.id_column and col != self.target_column:
                if df[col].nunique() == 1:
                    constant_features.append(col)
        
        # Drop constant features
        if constant_features:
            df = df.drop(columns=constant_features)
            dropped_info["constant_features"] = constant_features
            logger.info(f"Dropped constant features: {constant_features}")
        
        return