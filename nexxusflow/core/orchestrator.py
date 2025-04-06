class Orchestrator:
    def __init__(self, config):
        self.config = config

    def schedule_rerun(self, pipeline, interval_minutes=60):
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(lambda: pipeline.run({}), 'interval', minutes=interval_minutes)
        scheduler.start()
