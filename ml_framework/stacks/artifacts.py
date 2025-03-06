# 9. Storage Component
class StorageComponent(StackComponent):
    """
    Storage component for different backends (S3, GCS, local).
    """
    
    def initialize(self):
        """Initialize storage."""
        storage_type = self.config.get('type', 'local')
        
        if storage_type == 's3':
            try:
                import boto3
                self.client = boto3.client('s3')
                self.bucket = self.config.get('bucket')
            except ImportError:
                print("Warning: boto3 not installed. S3 storage disabled.")
                storage_type = 'local'
                
        elif storage_type == 'gcs':
            try:
                from google.cloud import storage
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.config.get('bucket'))
            except ImportError:
                print("Warning: google-cloud-storage not installed. GCS storage disabled.")
                storage_type = 'local'
                
        self.storage_type = storage_type
        
    def save(self, data, path):
        """Save data to storage."""
        if self.storage_type == 'local':
            import os
            import pickle
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
                
        elif self.storage_type == 's3':
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as temp:
                pickle.dump(data, temp)
                temp.flush()
                self.client.upload_file(temp.name, self.bucket, path)
                
        elif self.storage_type == 'gcs':
            import pickle
            
            blob = self.bucket.blob(path)
            blob.upload_from_string(pickle.dumps(data))
            
    def load(self, path):
        """Load data from storage."""
        if self.storage_type == 'local':
            import pickle
            
            with open(path, 'rb') as f:
                return pickle.load(f)
                
        elif self.storage_type == 's3':
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as temp:
                self.client.download_file(self.bucket, path, temp.name)
                with open(temp.name, 'rb') as f:
                    return pickle.load(f)
                    
        elif self.storage_type == 'gcs':
            import pickle
            
            blob = self.bucket.blob(path)
            return pickle.loads(blob.download_as_bytes())



# Artifact Store Component
class ArtifactStore(StackComponent):
    """Storage component for artifacts."""
    
    def initialize(self):
        """Initialize the artifact store."""
        store_type = self.config.get('type', 'local')
        base_path = self.config.get('path', './artifacts')
        
        if store_type == 's3':
            try:
                import boto3
                self.client = boto3.client('s3')
                self.bucket = self.config.get('bucket')
            except ImportError:
                print("Warning: boto3 not installed. Using local artifact store.")
                store_type = 'local'
                
        elif store_type == 'gcs':
            try:
                from google.cloud import storage
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.config.get('bucket'))
            except ImportError:
                print("Warning: google-cloud-storage not installed. Using local artifact store.")
                store_type = 'local'
                
        self.store_type = store_type
        self.base_path = base_path
        
        # Create local directory if needed
        if store_type == 'local' and not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            
    def save_artifact(self, artifact, name):
        """Save an artifact."""
        if self.store_type == 'local':
            import pickle
            path = os.path.join(self.base_path, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(artifact, f)
                
            return path
            
        elif self.store_type == 's3':
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as temp:
                pickle.dump(artifact, temp)
                temp.flush()
                path = os.path.join(self.base_path, name)
                self.client.upload_file(temp.name, self.bucket, path)
                
            return f"s3://{self.bucket}/{path}"
            
        elif self.store_type == 'gcs':
            import pickle
            
            path = os.path.join(self.base_path, name)
            blob = self.bucket.blob(path)
            blob.upload_from_string(pickle.dumps(artifact))
            
            return f"gs://{self.bucket.name}/{path}"
            
    def load_artifact(self, name):
        """Load an artifact."""
        if self.store_type == 'local':
            import pickle
            path = os.path.join(self.base_path, name)
            
            with open(path, 'rb') as f:
                return pickle.load(f)
                
        elif self.store_type == 's3':
            import pickle
            import tempfile
            
            path = os.path.join(self.base_path, name)
            with tempfile.NamedTemporaryFile() as temp:
                self.client.download_file(self.bucket, path, temp.name)
                with open(temp.name, 'rb') as f:
                    return pickle.load(f)
                    
        elif self.store_type == 'gcs':
            import pickle
            
            path = os.path.join(self.base_path, name)
            blob = self.bucket.blob(path)
            
            return pickle.loads(blob.download_as_bytes())

