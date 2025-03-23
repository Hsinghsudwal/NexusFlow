import prometheus_client
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

class Monitoring:
    def __init__(self, config):
        self.config = config
        self.report = Report(metrics=[DatasetDriftMetric()])

    def generate_report(self, reference_data, current_data):
        self.report.run(reference_data=reference_data, current_data=current_data)
        self.report.save_html(self.config['monitoring']['report_path'])