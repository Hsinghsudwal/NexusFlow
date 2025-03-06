# 11. File-based datasets
class CSVDataset(Dataset):
    """Dataset for CSV files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        """Load data from CSV."""
        import pandas as pd
        return pd.read_csv(self.filepath)
        
    def save(self, data):
        """Save data to CSV."""
        import os
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data.to_csv(self.filepath, index=False)


class ParquetDataset(Dataset):
    """Dataset for Parquet files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        """Load data from Parquet."""
        import pandas as pd
        return pd.read_parquet(self.filepath)
        
    def save(self, data):
        """Save data to Parquet."""
        import os
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data.to_parquet(self.filepath, index=False)


class PickleDataset(Dataset):
    """Dataset for Pickle files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        """Load data from Pickle."""
        import pickle
        with open(self.filepath, 'rb') as f:
            return pickle.load(f)
            
    def save(self, data):
        """Save data to Pickle."""
        import os
        import pickle
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)