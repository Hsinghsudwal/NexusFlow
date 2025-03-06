#!/usr/bin/env python
# cli.py
import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

from datapipe import ProjectContext

def setup_logger(log_level):
    """Set up logger with appropriate level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run data pipelines")
    
    parser.add_argument(
        "--project-path", "-p",
        type=str,
        default=os.getcwd(),
        help="Path to the project directory"
    )
    
    parser.add_argument(
        "--module", "-m",
        type=str,
        required=True,
        help="Module containing the pipeline to run"
    )
    
    parser.add_argument(
        "--pipeline", "-n",
        type=str,
        default="pipeline",
        help="Name of the pipeline variable in the module"
    )
    
    parser.add_argument(
        "--from-nodes",
        type=str,
        nargs="+",
        help="Run from these nodes onwards"
    )
    
    parser.add_argument(
        "--to-nodes",
        type=str,
        nargs="+",
        help="Run until these nodes"
    )
    
    parser.add_argument(
        "--only-nodes",
        type=str,
        nargs="+",
        help="Run only these nodes"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Run only nodes with these tags"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level"
    )
    
    parser.add_argument(
        "--create-project",
        action="store_true",
        help="Create a new project at the specified path"
    )
    
    return parser.parse_args()

def main():
    """Main CLI entry point."""
    args = parse_args()
    setup_logger(args.log_level)
    
    if args.create_project:
        from datapipe import create_project
        create_project(Path(args.project_path))
        return
    
    # Import the module dynamically
    try:
        project_path = Path(args.project_path)
        if project_path not in sys.path:
            sys.path.append(str(project_path))
        
        context = ProjectContext(project_path)
        
        # Run the pipeline
        kwargs = {}
        if args.from_nodes:
            kwargs["from_nodes"] = args.from_nodes
        if args.to_nodes:
            kwargs["to_nodes"] = args.to_nodes
        if args.only_nodes:
            kwargs["only_nodes"] = args.only_nodes
        if args.tags:
            kwargs["tags"] = args.tags
            
        results = context.run_pipeline(args.module, args.pipeline, **kwargs)
        
        # Log summary
        logging.info(f"Pipeline executed successfully with {len(results)} outputs")
        
    except Exception as e:
        logging.error(f"Failed to run pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()





# bash
# Create a new project
# python cli.py --create-project --project-path my_new_project

# Run a pipeline
# python cli.py -m src.pipelines.example --only-nodes load_data process_data