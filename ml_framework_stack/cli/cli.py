# 13. CLI Example
def create_cli():
    """Create a CLI for the framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="Pipeline to run")
    run_parser.add_argument("--env", default="local", help="Environment to run in")
    
    # List pipelines command
    list_parser = subparsers.add_parser("list", help="List available pipelines")
    
    # Create project command
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("name", help="Project name")
    
    return parser



# 12. CLI commands
def create_cli():
    """Create a CLI for the framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZenML-like Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="Pipeline to run")
    run_parser.add_argument("--stack", help="Stack to use")
    
    # Stack commands
    stack_parser = subparsers.add_parser("stack", help="Stack operations")
    stack_subparsers = stack_parser.add_subparsers(dest="stack_command")
    
    # Register stack
    register_parser = stack_subparsers.add_parser("register", help="Register a stack")
    register_parser.add_argument("config", help="Stack configuration file")
    
    # List stacks
    list_parser = stack_subparsers.add_parser("list", help="List registered stacks")
    
    # Set active stack
    activate_parser = stack_subparsers.add_parser("activate", help="Set the active stack")
    activate_parser.add_argument("name", help="Stack name")
    
    return parser