class FeatureValidationPipeline:
    def __init__(self, config):
        self.feature_store = FeatureStoreManager(config)
        self.validator = FeatureValidator(config)
        
    def validate_new_features(self, feature_df):
        validation_result = self.validator.validate_feature_pipeline(feature_df)
        if validation_result["valid"]:
            self.feature_store.write_online_features(feature_df)
        return validation_result
    
    def backfill_validation(self, start_date, end_date):
        historical_data = self.feature_store.get_historical_features(start_date, end_date)
        return self.validator.validate_feature_pipeline(historical_data)