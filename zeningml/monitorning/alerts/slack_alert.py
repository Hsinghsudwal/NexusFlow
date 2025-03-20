from slack_sdk import WebClient

class SlackAlerter:
    def __init__(self, config):
        self.client = WebClient(token=config.get("slack.token"))
        self.channel = config.get("slack.channel")
        
    def send_alert(self, message):
        self.client.chat_postMessage(
            channel=self.channel,
            text=f":warning: MLOps Alert: {message}"
        )