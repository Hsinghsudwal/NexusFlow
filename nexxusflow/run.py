# main.py
import sys
import argparse
from core.config import load_config
from pipelines.training_pipeline import training_pipeline
from rollback.rollback_manager import RollbackManager
from feedback.feedback_loop import FeedbackLoop
from scheduler.scheduler import schedule_pipeline


def run_pipeline(args):
    config = load_config(args.mode)
    if args.pipeline == 'training':
        print(f"Running training pipeline in {args.mode} mode")
        training_pipeline(config)


def run_rollback(args):
    config = load_config(args.mode)
    manager = RollbackManager(config)
    model = manager.rollback(args.model_key)
    print(f"Rolled back to model: {model}")


def collect_feedback(args):
    config = load_config(args.mode)
    feedback = FeedbackLoop(config)
    feedback.collect_feedback(args.data)
    print("Feedback collected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['local', 'cloud', 'localstack'], required=True)
    parser.add_argument('--pipeline', choices=['training'], default='training')
    parser.add_argument('--action', choices=['run', 'rollback', 'feedback', 'schedule'], required=True)
    parser.add_argument('--model_key', help='Model key to rollback')
    parser.add_argument('--data', help='Feedback data in stringified JSON')
    parser.add_argument('--interval', type=int, help='Schedule interval in minutes')
    args = parser.parse_args()

    if args.action == 'run':
        run_pipeline(args)
    elif args.action == 'rollback':
        run_rollback(args)
    elif args.action == 'feedback':
        collect_feedback(args)
    elif args.action == 'schedule':
        config = load_config(args.mode)
        schedule_pipeline(lambda: training_pipeline(config), args.interval)


if __name__ == '__main__':
    main()
