import os
import boto3

class ArtifactManager:
    def __init__(self, config):
        self.mode = config['mode']
        self.bucket = config['artifact_store']['bucket']
        self.local_path = config['artifact_store'].get('local_path', './artifacts')

        if self.mode in ['cloud', 'localstack']:
            self.s3 = boto3.client('s3', endpoint_url=config.get('s3_endpoint'))

    def save(self, key, data):
        if self.mode == 'local':
            path = os.path.join(self.local_path, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(data)
        else:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data.encode())

    def load(self, key):
        if self.mode == 'local':
            path = os.path.join(self.local_path, key)
            with open(path, 'r') as f:
                return f.read()
        else:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return obj['Body'].read().decode()
