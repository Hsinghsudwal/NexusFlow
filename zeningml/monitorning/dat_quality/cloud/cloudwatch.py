import boto3

class CloudWatchMetrics:
    def __init__(self, config):
        self.client = boto3.client('cloudwatch',
            aws_access_key_id=config.get('aws.access_key'),
            aws_secret_access_key=config.get('aws.secret_key'))
        
    def put_metric(self, namespace, metric_name, value):
        self.client.put_metric_data(
            Namespace=namespace,
            MetricData=[{
                'MetricName': metric_name,
                'Value': value,
                'Timestamp': datetime.now()
            }]
        )

class DatadogMonitor:
    def __init__(self, config):
        from datadog import initialize, api
        initialize(**config.get('datadog'))
        self.api = api
        
    def send_metric(self, metric_name, value):
        self.api.Metric.send(
            metric=metric_name,
            points=[(datetime.now().timestamp(), value)]
        )