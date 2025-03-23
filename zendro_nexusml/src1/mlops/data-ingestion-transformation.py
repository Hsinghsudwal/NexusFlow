# src/data/ingestion.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Data ingestion component that extracts data from various sources
    and prepares it for processing.
    """
    
    def __init__(self, config):
        """
        Initialize the data ingestion component.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.db_connection = None
        self.s3_client = None
        
        # Initialize connections
        self._init_db_connection()
        self._init_s3_client()
        
    def _init_db_connection(self):
        """Initialize database connection."""
        try:
            conn_string = f"postgresql://{self.config['postgres_user']}:{self.config['postgres_password']}@{self.config['postgres_host']}:{self.config['postgres_port']}/{self.config['postgres_db']}"
            self.db_connection = create_engine(conn_string)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
            
    def _init_s3_client(self):
        """Initialize S3 client (using MinIO or LocalStack in development)."""
        try:
            endpoint_url = self.config.get('s3_endpoint_url', None)
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=self.config['aws_access_key_id'],
                aws_secret_access_key=self.config['aws_secret_access_key'],
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            logger.info("S3 client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def extract_from_database(self, query):
        """
        Extract data from database using SQL query.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            logger.info(f"Extracting data with query: {query}")
            df = pd.read_sql(query, self.db_connection)
            logger.info(f"Extracted {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data from database: {str(e)}")
            raise
    
    def extract_from_s3(self, bucket, key):
        """
        Extract data from S3 bucket.
        
        Args:
            bucket (str): S3 bucket name
            key (str): S3 object key
            
        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            logger.info(f"Extracting data from S3: {bucket}/{key}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            if key.endswith('.csv'):
                df = pd.read_csv(response['Body'])
            elif key.endswith('.parquet'):
                df = pd.read_parquet(response['Body'])
            elif key.endswith('.json'):
                df = pd.read_json(response['Body'])
            else:
                raise ValueError(f"Unsupported file format for key: {key}")
                
            logger.info(f"Extracted {len(df)} records from S3")
            return df
        except ClientError as e:
            logger.error(f"Failed to extract data from S3: {str(e)}")
            raise
    
    def save_to_processed(self, df, filename):
        """
        Save extracted data to processed directory.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Filename to save data
            
        Returns:
            str: Path where data is saved
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_dir = self.config['processed_data_dir']
            
            # Create directory if it doesn't exist
            os.makedirs(processed_dir, exist_ok=True)
            
            # Save data
            file_path = os.path.join(processed_dir, f"{filename}_{timestamp}.parquet")
            df.to_parquet(file_path, index=False)
            
            logger.info(f"Saved processed data to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            raise

    def extract_customer_data(self):
        """
        Extract customer churn data from all configured sources.
        
        Returns:
            pd.DataFrame: Combined customer data
        """
        # Extract customer profile data
        customer_query = """
        SELECT 
            customer_id, 
            age, 
            gender, 
            tenure, 
            contract_type,
            monthly_charges, 
            total_charges,
            has_phone_service,
            has_multiple_lines,
            has_internet_service,
            has_online_security,
            has_tech_support,
            churned
        FROM customer_profiles
        """
        customer_df = self.extract_from_database(customer_query)
        
        # Extract customer usage data from S3
        try:
            usage_df = self.extract_from_s3(
                self.config['usage_data_bucket'],
                self.config['usage_data_key']
            )
            
            # Merge customer profile and usage data
            merged_df = pd.merge(
                customer_df, 
                usage_df,
                on='customer_id',
                how='left'
            )
            
            # Save processed data
            self.save_to_processed(merged_df, 'customer_churn_data')
            
            return merged_df
        except Exception as e:
            logger.error(f"Failed to extract and process customer data: {str(e)}")
            # If usage data extraction fails, return only customer profile data
            logger.warning("Returning only customer profile data")
            self.save_to_processed(customer_df, 'customer_churn_data_profiles_only')
            return customer_df


# src/data/transformation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTransformation:
    """
    Data transformation component for preprocessing and feature engineering.
    """
    
    def __init__(self, config):
        """
        Initialize the data transformation component.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.preprocessing_pipeline = None
        
    def _build_preprocessing_pipeline(self, df):
        """
        Build the preprocessing pipeline for numerical and categorical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame to determine feature types
            
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target variable from feature lists if present
        target_col = self.config['target_column']
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'  # Drop columns not specified
        )
        
        logger.info(f"Built preprocessing pipeline with {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
        
        return preprocessor
    
    def transform(self, df):
        """
        Transform the input data using the preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input data to transform
            
        Returns:
            tuple: (X, y) where X is the transformed features and y is the target
        """
        # Build preprocessing pipeline if not already built
        if self.preprocessing_pipeline is None:
            self.preprocessing_pipeline = self._build_preprocessing_pipeline(df)
        
        # Separate features and target
        X = df.drop(columns=[self.config['target_column']])
        y = df[self.config['target_column']]
        
        # Transform features
        X_transformed = self.preprocessing_pipeline.fit_transform(X)
        
        logger.info(f"Transformed data with shape: {X_transformed.shape}")
        
        return X_transformed, y
    
    def save_pipeline(self):
        """
        Save the preprocessing pipeline to disk.
        
        Returns:
            str: Path where pipeline is saved
        """
        if self.preprocessing_pipeline is None:
            logger.error("Cannot save preprocessing pipeline: not initialized")
            raise ValueError("Preprocessing pipeline is not initialized")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.config['model_dir']
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save pipeline
        pipeline_path = os.path.join(model_dir, f"preprocessing_pipeline_{timestamp}.joblib")
        joblib.dump(self.preprocessing_pipeline, pipeline_path)
        
        logger.info(f"Saved preprocessing pipeline to {pipeline_path}")
        return pipeline_path
    
    def load_pipeline(self, pipeline_path):
        """
        Load a preprocessing pipeline from disk.
        
        Args:
            pipeline_path (str): Path to the saved pipeline
            
        Returns:
            ColumnTransformer: Loaded preprocessing pipeline
        """
        try:
            self.preprocessing_pipeline = joblib.load(pipeline_path)
            logger.info(f"Loaded preprocessing pipeline from {pipeline_path}")
            return self.preprocessing_pipeline
        except Exception as e:
            logger.error(f"Failed to load preprocessing pipeline: {str(e)}")
            raise
    
    def engineer_features(self, df):
        """
        Create new features from existing data.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        # Make a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        try:
            # Example feature engineering - customize for your use case
            if 'tenure' in df.columns and 'monthly_charges' in df.columns:
                # Customer lifetime value (simplified)
                enhanced_df['customer_lifetime_value'] = df['tenure'] * df['monthly_charges']
            
            if 'contract_type' in df.columns:
                # Contract risk factor (higher for month-to-month)
                contract_risk = {
                    'Month-to-month': 3,
                    'One year': 2,
                    'Two year': 1
                }
                enhanced_df['contract_risk'] = df['contract_type'].map(contract_risk)
            
            # Add service count feature
            service_columns = [col for col in df.columns if col.startswith('has_')]
            if service_columns:
                enhanced_df['service_count'] = df[service_columns].sum(axis=1)
            
            logger.info(f"Engineered features: {list(set(enhanced_df.columns) - set(df.columns))}")
            
            return enhanced_df
        except Exception as e:
            logger.error(f"Failed during feature engineering: {str(e)}")
            # If feature engineering fails, return original data
            logger.warning("Returning original data without engineered features")
            return df
