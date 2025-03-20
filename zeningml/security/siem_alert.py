class SIEMIntegrator:
    def __init__(self, config):
        self.endpoint = config['siem']['endpoint']
        self.auth = (config['siem']['user'], config['siem']['pass'])
        
    def format_cef(self, alert):
        return f"CEF:0|MLOps|{alert['component']}|1.0|{alert['code']}|{alert['message']}|5|"

    def send_alert(self, alert):
        requests.post(
            self.endpoint,
            data=self.format_cef(alert),
            auth=self.auth
        )

class SplunkAlert(SIEMIntegrator):
    def format_hec(self, alert):
        return {
            "event": alert,
            "sourcetype": "mlops:alert",
            "index": "ml_security"
        }