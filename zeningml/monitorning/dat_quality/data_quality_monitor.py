class DataQualityMonitor:
    def __init__(self, config):
        self.validator = DataValidator(config)
        self.quality_metrics = {}
        
    def continuous_validation(self, data_stream):
        for batch in data_stream:
            result = self.validator.validate(batch)
            self._update_quality_metrics(result)
            if not result["valid"]:
                self._trigger_data_alert(result)
                
    def _update_quality_metrics(self, result):
        for expectation in result["results"]:
            metric_name = f"data_quality.{expectation['expectation_type']}"
            self.quality_metrics[metric_name] = expectation['success']

class FeatureValidator(DataQualityMonitor):
    def validate_feature_pipeline(self, feature_df):
        return self.validator.validate(feature_df)