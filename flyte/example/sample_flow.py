import argparse
from nexusml import Workflow, Task

def main():
    parser = argparse.ArgumentParser(description='Run NexusML workflows')
    parser.add_argument('workflow', help='The name of the workflow to execute')
    args = parser.parse_args()

    # Example to load and run the workflow
    workflow = Workflow(name=args.workflow)
    # Add tasks to the workflow
    # workflow.add_task(task)
   
    workflow.run()

if __name__ == '__main__':
    main()