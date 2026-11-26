import shutil
from pathlib import Path

from invoke import task

from .system import DOCS_BUILD_DIR


@task
def clean(context):
    """Remove the documentation build."""
    shutil.rmtree(DOCS_BUILD_DIR, ignore_errors=True)


@task
def build(context):
    """Build the documentation and export the example notebooks."""
    context.run("zensical build")
    for notebook in sorted(Path("notebooks").glob("*.py")):
        destination = Path(DOCS_BUILD_DIR) / notebook.with_suffix(".html").name
        context.run(
            f'marimo export html "{notebook}" -o "{destination}" --force'
        )


@task(default=True)
def serve(context):
    """Run the Zensical development server."""
    context.run(f"zensical serve --dev-addr localhost:{context.start_port + 1}")
