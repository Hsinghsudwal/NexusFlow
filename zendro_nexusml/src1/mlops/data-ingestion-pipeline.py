import os
import logging
import pandas as pd
from datetime import datetime
from prefect import task

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Pipeline for ingesting data from various sources
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.raw_data_path = config["data"]["raw_data_path"]
        
    def run(self):
        """Run the data ingestion pipeline"""
        logger.info("Starting data ingestion")
        
        # Check if we have a local file
        if os.path.exists(self.raw_data_path):
            logger.info(f"Loading data from local file: {self.raw_data_path}")
            return self._load_data_from_file()
        
        # If no local file, generate sample data for demo purposes
        logger.info("No local data found, generating sample data")
        return self._generate_sample_data()
        
    def _load_data_from_file(self):
        """Load data from a local file"""
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded {len(df)} rows from {self.raw_data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from file: {str(e)}")
            raise
            
    def _generate_sample_data(self):
        """Generate sample customer churn data for demonstration"""
        logger.info("Generating sample customer churn data")
        
        # Sample size
        n_samples = 1000
        
        # Create random data
        import numpy as np
        
        # Set a seed for reproducibility
        np.random.seed(42)
        
        # Generate customer IDs
        customer_ids = [f"CUST{i:06d}" for i in range(1, n_samples + 1)]
        
        # Generate features
        genders = np.random.choice(['Male', 'Female'], size=n_samples)
        
        tenures = np.round(np.random.gamma(shape=2.0, scale=18.0, size=n_samples))
        tenures = np.clip(tenures, 1, 72)  # Clip to 1-72 months
        
        contract_types = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            size=n_samples, 
            p=[0.6, 0.25, 0.15]  # Month-to-month more common
        )
        
        payment_methods = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
            size=n_samples
        )
        
        # Monthly charges: higher for month-to-month contracts
        monthly_charges = np.zeros(n_samples)
        for i, contract in enumerate(contract_types):
            if contract == 'Month-to-month':
                monthly_charges[i] = np.random.uniform(20, 120)
            elif contract == 'One year':
                monthly_charges[i] = np.random.uniform(30, 90)
            else:  # Two year
                monthly_charges[i] = np.random.uniform(20, 80)
        monthly_charges = np.round(monthly_charges, 2)
        
        # Total charges: monthly charges * tenure with some variation
        total_charges = monthly_charges * tenures * np.random.uniform(0.95, 1.05, size=n_samples)
        total_charges = np.round(total_charges, 2)
        
        # Churn is influenced by contract type and tenure
        churn_prob = np.zeros(n_samples)
        for i in range(n_samples):
            # Base probability based on contract type
            if contract_types[i] == 'Month-to-month':
                base_prob = 0.4
            elif contract_types[i] == 'One year':
                base_prob = 0.2
            else:  # Two year
                base_prob = 0.1
                
            # Adjust based on tenure (longer tenure reduces churn)
            tenure_factor = np.clip(1.0 - (tenures[i] / 72) * 0.8, 0.3, 1.0)
            
            # Adjust based on monthly charges (higher charges increase churn slightly)
            charge_factor = np.clip(monthly_charges[i] / 120, 0.8, 1.2)
            
            # Compute final probability
            churn_prob[i] = base_prob * tenure_factor * charge_factor
        
        # Generate churn labels
        churn = np.random.binomial(1, churn_prob)
        churn = ['Yes' if c == 1 else 'No' for c in churn]
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'gender': genders,
            'tenure': tenures,
            'contract_type': contract_types,
            'payment_method': payment_methods,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'churn': churn
        })
        
        # Save the generated data to the raw data path
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        df.to_csv(self.raw_data_path, index=False)
        logger.info(f"Saved {len(df)} rows of sample data to {self.raw_data_path}")
        
        return df
