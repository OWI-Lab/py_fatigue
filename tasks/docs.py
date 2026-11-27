import re
import shutil
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from textwrap import dedent

from invoke import Context, task

from .system import DOCS_BUILD_DIR


@task
def clean(context):
    """Remove the documentation build."""
    shutil.rmtree(DOCS_BUILD_DIR, ignore_errors=True)


@task
def getting_started(context: Context) -> None:
    """Export the Getting Started guide and its runnable notebook source."""
    guide = Path("notebooks/guides/getting-started.md").read_text(
        encoding="utf-8"
    )
    guide = re.sub(r"^```[ \t]*python[ \t]*$", "```{python}", guide, flags=re.M)
    guide = re.sub(r"(\]\([^():]+)\.md(?=[#)])", r"\1.html", guide)
    guide = re.sub(
        r'^=== "([^"]+)"\n\n((?:^    [^\n]*\n)+)',
        lambda match: f"**{match[1]}**\n\n{dedent(match[2])}",
        guide,
        flags=re.M,
    )
    destination = Path(DOCS_BUILD_DIR) / "user/getting-started-notebook.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stylesheet = Path(DOCS_BUILD_DIR) / "stylesheets/custom.css"
    stylesheet.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("docs/stylesheets/custom.css", stylesheet)
    with TemporaryDirectory() as temporary:
        markdown = Path(temporary) / "getting-started.md"
        markdown.write_text(
            "```{python}\nimport marimo as mo\n"
            "from notebooks._examples.charts import plot_views\n"
            "from notebooks._examples.synthetic import display_indices\n"
            "```\n\n" + guide,
            encoding="utf-8",
        )
        context.run(
            f"marimo convert {quote(str(markdown))} -o {quote(str(destination))}"
        )
    source = destination.read_text(encoding="utf-8")
    source = source.replace(
        "app = marimo.App()",
        'app = marimo.App(width="full", css_file="../stylesheets/custom.css")',
        1,
    )
    source = source.replace(
        "@app.cell\ndef _():", "@app.cell(hide_code=True)\ndef _():", 1
    )
    destination.write_text(source, encoding="utf-8")
    context.run(
        f"marimo check --fix --unsafe-fixes --select MF004 "
        f"{quote(str(destination))}"
    )
    context.run(
        f"marimo export html {quote(str(destination))} "
        f"-o {quote(str(destination.with_suffix('.html')))} --force"
    )


@task
def build(context: Context) -> None:
    """Build the documentation and export the example notebooks."""
    context.run("zensical build --clean")
    getting_started(context)
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
