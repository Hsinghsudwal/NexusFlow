import boto3
from core.artifact_store import ArtifactStore

class S3ArtifactStore(ArtifactStore):
    def __init__(self, config):
        super().__init__(config)
        self.bucket_name = config.get("aws", {}).get("bucket_name")
        self.s3 = boto3.client('s3',
            aws_access_key_id=config.get("aws", {}).get("access_key"),
            aws_secret_access_key=config.get("aws", {}).get("secret_key"))

    def save_artifact(self, artifact: Any, subdir: str, name: str) -> None:
        local_path = f"{self.base_path}/{subdir}/{name}"
        super().save_artifact(artifact, subdir, name)
        s3_path = f"{subdir}/{name}"
        self.s3.upload_file(local_path, self.bucket_name, s3_path)



from .localstack.localstack_client import LocalStackClient

class S3ArtifactStore(ArtifactStore):
    def __init__(self, config):
        super().__init__(config)
        
        if config.get("aws.use_localstack"):
            self.client = LocalStackClient(config).get_client("s3")
            self.resource = LocalStackClient(config).get_resource("s3")
            initialize_localstack(config)
            check_localstack_connection(config)
        else:
            # Regular AWS client initialization
            self.client = boto3.client(
                "s3",
                aws_access_key_id=config.get("aws.access_key"),
                aws_secret_access_key=config.get("aws.secret_key"),
                region_name=config.get("aws.region")
            )

        self.bucket_name = config.get("aws.s3.bucket_name")



