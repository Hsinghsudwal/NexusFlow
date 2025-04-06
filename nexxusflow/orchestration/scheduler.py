# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler

def schedule_pipeline(pipeline_fn, interval_minutes):
    scheduler = BackgroundScheduler()
    scheduler.add_job(pipeline_fn, 'interval', minutes=interval_minutes)
    scheduler.start()
    print(f"Scheduled {pipeline_fn.__name__} every {interval_minutes} minutes")
