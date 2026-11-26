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
    notebook_root = Path("notebooks")
    notebooks = sorted(notebook_root.glob("*.py"))
    notebooks.extend(sorted(notebook_root.glob("tutorials/[0-9][0-9]-*.py")))
    for notebook in notebooks:
        relative_html = notebook.relative_to(notebook_root).with_suffix(".html")
        destination = Path(DOCS_BUILD_DIR) / relative_html
        destination.parent.mkdir(parents=True, exist_ok=True)
        context.run(
            f'marimo export html "{notebook}" -o "{destination}" --force'
        )


@task(default=True)
def serve(context):
    """Run the Zensical development server."""
    context.run(f"zensical serve --dev-addr localhost:{context.start_port + 1}")
