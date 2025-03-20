import great_expectations as ge
from typing import Dict

class DataValidator:
    def __init__(self, expectation_suite: Dict):
        self.suite = ge.core.ExpectationSuite(expectation_suite)
        
    def validate(self, dataframe):
        results = ge.dataset.PandasDataset(dataframe).validate(
            expectation_suite=self.suite,
            result_format="SUMMARY"
        )
        return {
            'valid': results['success'],
            'results': results['results']
        }

    def generate_validation_report(self, validation_result):
        return ge.render.view.DefaultJinjaPageView().render(
            ge.render.RenderedContent(
                **validation_result.to_json_dict()
            )
        )


    # validation/expectation_suites/taxi_suite.json
{
  "expectations": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "fare_amount"}
    },
    {
      "expectation_type": "expect_column_values_to_be_between",
      "kwargs": {"column": "passenger_count", "min_value": 1, "max_value": 8}
    }
  ]
}