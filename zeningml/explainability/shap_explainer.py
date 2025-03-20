import shap
import mlflow

class SHAPExplainer:
    def __init__(self, model, training_data):
        self.explainer = shap.Explainer(model, training_data)
        
    def explain_instance(self, instance):
        shap_values = self.explainer(instance)
        return {
            "base_value": shap_values.base_values,
            "values": shap_values.values,
            "feature_names": shap_values.feature_names
        }
    
    def log_explanation(self, explanation):
        mlflow.log_dict(explanation, "shap_explanation.json")
        shap.plots.waterfall(explanation, show=False)
        mlflow.log_figure(plt.gcf(), "shap_waterfall.png")