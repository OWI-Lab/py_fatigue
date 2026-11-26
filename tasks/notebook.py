from invoke import task


@task(default=True, aliases=("start",))
def run(context):
    """Open the marimo editor for project notebooks."""
    context.run(f"marimo edit notebooks --port {context.start_port}")
