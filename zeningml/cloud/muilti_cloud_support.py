class CloudProviderFactory:
    @staticmethod
    def get_client(service_type, config):
        if config['cloud'] == 'aws':
            return AWSClient(service_type, config)
        elif config['cloud'] == 'gcp':
            return GCPClient(service_type, config)
        elif config['cloud'] == 'azure':
            return AzureClient(service_type, config)

class MultiCloudArtifactStore:
    def __init__(self, config):
        self.client = CloudProviderFactory.get_client('storage', config)
        
    def save_model(self, model, path):
        self.client.upload(
            model.serialize(), 
            f"{path}/{model.version}"
        )