from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ModelPerformanceTab
from prefect import task
import logging

logger = logging.getLogger(__name__)

class ModelMonitoring:
    @task
    def setup_monitoring(self, reference_data: pd.DataFrame, config: Dict) -> str:
        """Set up monitoring dashboard."""
        logger.info("Setting up model monitoring")

        try:
            dashboard = Dashboard(tabs=[DataDriftTab(), ModelPerformanceTab()])
            dashboard_path = config['monitoring']['dashboard_path']
            dashboard.save(dashboard_path)
            return dashboard_path

        except Exception as e:
            logger.error(f"Error during monitoring setup: {e}")
            raise