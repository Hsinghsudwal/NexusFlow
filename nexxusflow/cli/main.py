import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument("pipeline_file")
def run(pipeline_file):
    """Execute a pipeline using active stack"""
    from core.pipeline import Pipeline
    pipeline = Pipeline.load(pipeline_file)
    pipeline.run()