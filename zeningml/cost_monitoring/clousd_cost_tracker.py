from abc import ABC, abstractmethod

class CloudCostMonitor(ABC):
    @abstractmethod
    def get_current_cost(self):
        pass

class AWSCostMonitor(CloudCostMonitor):
    def __init__(self, config):
        self.client = boto3.client('ce')
        
    def get_current_cost(self):
        return self.client.get_cost_and_usage(
            TimePeriod={
                'Start': str(date.today()),
                'End': str(date.today())
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost']
        )

class ModelCostCalculator:
    def __init__(self, model):
        self.model = model
        
    def calculate_inference_cost(self, num_requests):
        return {
            'compute_cost': self.model.compute_cost * num_requests,
            'memory_cost': self.model.memory_usage * 0.001,
            'total': self.model.compute_cost * num_requests + self.model.memory_usage * 0.001
        }