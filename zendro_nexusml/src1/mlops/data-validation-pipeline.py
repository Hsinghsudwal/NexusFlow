import logging
import pandas as pd
import numpy as np
from prefect import task

logger = logging.getLogger(__name__)

class DataValidationPipeline:
    """
    Pipeline for validating data quality and integrity
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.target_column = config["data"]["target_column"]
        self.id_column = config["data"]["id_column"]
        
    def run(self, data):
        """Run the data validation pipeline"""
        logger.info("Starting data validation")
        
        # Perform validation checks
        validation_report = {}
        validation_report["missing_values"] = self._check_missing_values(data)
        validation_report["duplicates"] = self._check_duplicates(data)
        validation_report["data_types"] = self._check_data_types(data)
        validation_report["target_distribution"] = self._check_target_distribution(data)
        validation_report["outliers"] = self._check_outliers(data)
        
        # Compile summary
        validation_report["summary"] = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "missing_values_count": validation_report["missing_values"]["total_missing"],
            "duplicate_rows_count": validation_report["duplicates"]["duplicate_count"],
            "data_quality_score": self._calculate_quality_score(validation_report)
        }
        
        logger.info(f"Data validation complete: {validation_report['summary']}")
        
        # Perform any necessary data corrections
        corrected_data = self._correct_data(data, validation_report)
        
        return corrected_data, validation_report
    
    def _check_missing_values(self, data):
        """Check for missing values in the dataset"""
        missing_counts = data.isnull().sum().to_dict()
        missing_percentage = (data.isnull().mean() * 100).to_dict()
        total_missing = data.isnull().sum().sum()
        
        return {
            "column_missing_counts": missing_counts,
            "column_missing_percentage": missing_percentage,
            "total_missing": total_missing
        }
    
    def _check_duplicates(self, data):
        """Check for duplicate rows in the dataset"""
        duplicate_count = data.duplicated().sum()
        duplicate_ids = None
        
        if self.id_column in data.columns:
            # Check for duplicate IDs
            id_counts = data[self.id_column].value_counts()
            duplicate_ids = id_counts[id_counts > 1].to_dict()
        
        return {
            "duplicate_count": duplicate_count,
            "duplicate_ids": duplicate_ids
        }
    
    def _check_data_types(self, data):
        """Check data types of each column"""
        data_types = data.dtypes.astype(str).to_dict()
        
        # Check for numeric columns with mixed types (e.g., strings mixed with numbers)
        mixed_type_cols = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(data[col])
                except:
                    # If some values could be numeric but not all, flag as mixed
                    try:
                        numeric_mask = pd.to_numeric(data[col], errors='coerce').notnull()
                        if numeric_mask.sum() > 0 and numeric_mask.sum() < len(data):
                            mixed_type_cols[col] = f"{numeric_mask.sum()} numeric values, {len(data) - numeric_mask.sum()} non-numeric"
                    except:
                        pass
        
        return {
            "column_data_types": data_types,
            "mixed_type_columns": mixed_type_cols
        }
    
    def _check_target_distribution(self, data):
        """Check the distribution of the target variable"""
        if self.target_column not in data.columns:
            return {"error": f"Target column '{self.target_column}' not found"}
        
        target_counts = data[self.target_column].value_counts().to_dict()
        target_percentages = (data[self.target_column].value_counts(normalize=True) * 100).to_dict()
        
        # Check if binary classification is balanced
        balance_ratio = None
        if len(target_counts) == 2:
            min_class_count = min(target_counts.values())
            max_class_count = max(target_counts.values())
            balance_ratio = min_class_count / max_class_count
        
        return {
            "target_counts": target_counts,
            "target_percentages": target_percentages,
            "balance_ratio": balance_ratio
        }
    
    def _check_outliers(self, data):
        """Check for outliers in numeric columns"""
        outliers = {}
        
        for col in data.select_dtypes(include=['int64', 'float64']).columns:
            # Skip ID column
            if col == self.id_column:
                continue
                
            # Calculate Z-scores
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_count = (z_scores > 3).sum()
            
            # Calculate IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outlier_count > 0 or iqr_outlier_count > 0:
                outliers[col] = {
                    "z_score_outliers": int(outlier_count),
                    "iqr_outliers": int(iqr_outlier_count),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "mean": float(data[col].mean()),
                    "median": float(data[col].median())
                }
        
        return outliers
    
    def _calculate_quality_score(self, validation_report):
        """Calculate an overall data quality score (0-100)"""
        total_rows = validation_report["summary"]["total_rows"]
        total_columns = validation_report["summary"]["total_columns"]
        
        # Penalize for missing values
        missing_penalty = min(40, (validation_report["missing_values"]["total_missing"] / (total_rows * total_columns)) * 100)
        
        # Penalize for duplicates
        duplicate_penalty = min(20, (validation_report["duplicates"]["duplicate_count"] / total_rows) * 100)
        
        # Penalize for mixed data types
        mixed_type_penalty = min(20, len(validation_report["data_types"]["mixed_type_columns"]) * 5)
        
        # Penalize for severe class imbalance
        imbalance_penalty = 0
        if validation_report["target_distribution"].get("balance_ratio") is not None:
            balance_ratio = validation_report["target_distribution"]["balance_ratio"]
            if balance_ratio < 0.1:
                imbalance_penalty = 20
            elif balance_ratio < 0.25:
                imbalance_penalty = 10
            elif balance_ratio < 0.5:
                imbalance_penalty = 5
        
        # Calculate final score
        score = 100 - missing_penalty - duplicate_penalty - mixed_type_penalty - imbalance_penalty
        
        return max(0, min(100, score))
    
    def _correct_data(self, data, validation_report):
        """Apply corrections to the data based on validation findings"""
        df = data.copy()
        
        # Remove duplicates if any
        if validation_report["duplicates"]["duplicate_count"] > 0:
            logger.info(f"Removing {validation_report['duplicates']['duplicate_count']} duplicate rows")
            df = df.drop_duplicates()
        
        # Handle the mixed type columns if needed
        for col, info in validation_report["data_types"].get("mixed_type_columns", {}).items():
            logger.info(f"Attempting to convert mixed type column '{col}'")
            # For demonstration, convert to numeric and fill NaN with median
            if col != self.target_column and col != self.id_column:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().sum() > 0:
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    logger.info(f"Filled {col} NaN values with median: {median_value}")
        
        # Fill missing values in numeric columns with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col != self.id_column and df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled {col} NaN values with median: {median_value}")
        
        # Fill missing values in categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != self.id_column and df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled {col} NaN values with mode: {mode_value}")
        
        return df
