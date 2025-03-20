def initialize_localstack(config):
    """Initialize LocalStack resources based on config"""
    client = LocalStackClient(config).get_client("s3")
    
    if config.get("aws.s3.auto_create_bucket"):
        try:
            client.create_bucket(
                Bucket=config.get("aws.s3.bucket_name")
            )
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass

def check_localstack_connection(config):
    """Verify LocalStack connectivity"""
    try:
        client = LocalStackClient(config).get_client("s3")
        client.list_buckets()
        return True
    except Exception as e:
        raise ConnectionError(f"LocalStack connection failed: {str(e)}")