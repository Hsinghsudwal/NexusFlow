import boto3
from botocore.config import Config

class LocalStackClient:
    def __init__(self, config):
        self.endpoint = config.get("localstack.endpoint")
        self.region = config.get("aws.region")
        
        self.client_config = Config(
            region_name=self.region,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )

    def get_client(self, service_name):
        return boto3.client(
            service_name,
            endpoint_url=f"{self.endpoint}",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            config=self.client_config
        )

    def get_resource(self, service_name):
        return boto3.resource(
            service_name,
            endpoint_url=f"{self.endpoint}",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            config=self.client_config
        )