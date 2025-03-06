"""CLI command definitions."""
import click
from rich.console import Console
from rich.table import Table
from typing import Optional

from nexusml.tracking.experiment import Experiment
from nexusml.core.config import Config

console = Console()

@click.group()
def cli():
    """NexusML CLI interface."""
    pass

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str):
    """Run a pipeline from config file."""
    try:
        config = Config(config_path)
        # Pipeline execution logic here
        console.print(f"Pipeline executed successfully from {config_path}")
    except Exception as e:
        console.print(f"[red]Error running pipeline: {str(e)}[/red]")

@cli.command()
def list_experiments():
    """List all experiments."""
    experiments = Experiment.list_experiments()
    
    table = Table(title="Experiments")
    table.add_column("Pipeline ID")
    table.add_column("Pipeline Name")
    table.add_column("Start Time")
    table.add_column("End Time")
    table.add_column("Status")
    
    for exp in experiments:
        table.add_row(
            exp[0],  # pipeline_id
            exp[1],  # pipeline_name
            str(exp[2]),  # start_time
            str(exp[3]) if exp[3] else "-",  # end_time
            exp[4]  # status
        )
    
    console.print(table)

if __name__ == "__main__":
    cli()
