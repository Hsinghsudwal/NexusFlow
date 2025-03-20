import docker
import requests
from github import Github

class MLCICD:
    def __init__(self, config):
        self.config = config
        self.docker_client = docker.from_env()
        self.gh_client = Github(config.get('github_token'))

    def build_docker_image(self, dockerfile_path):
        image, _ = self.docker_client.images.build(
            path=dockerfile_path,
            tag=f"{self.config['image_name']}:{self.config['version']}"
        )
        return image

    def deploy_to_kubernetes(self, deployment_config):
        # Implement actual deployment logic
        return requests.post(
            self.config['k8s_endpoint'],
            json=deployment_config
        )

    def run_pipeline_tests(self):
        # Implement testing framework
        return subprocess.run(
            ["pytest", "tests/"],
            check=True
        )