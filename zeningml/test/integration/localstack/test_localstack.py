import pytest
from integrations.aws.localstack.localstack_setup import check_localstack_connection

class TestLocalStackIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, localstack_config):
        self.config = localstack_config

    def test_s3_connection(self):
        assert check_localstack_connection(self.config) == True

    def test_artifact_store_operations(self):
        store = S3ArtifactStore(self.config)
        test_data = {"test": "data"}
        
        # Test write/read operations
        store.save_artifact(test_data, "test", "test_artifact.pkl")
        retrieved = store.load_artifact("test", "test_artifact.pkl")
        assert retrieved == test_data