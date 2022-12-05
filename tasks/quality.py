from invoke import task
from .colors import colorize

@task
def black(c):
    """Run code formatter: black."""
    print(colorize("-> Running black..."))
    c.run(f"poetry run black {c.project_slug}", pty=True)


@task
def flake(c):
    """Run style guide enforcement: flake8."""
    print(colorize("-> Running flake8..."))
    c.run(f"poetry run flake8 {c.project_slug}", warn=True, pty=True)


@task
def pylint(c):
    """Run code analysis: pylint."""
    print(colorize("-> Running pylint..."))
    c.run(f"poetry run pylint {c.project_slug}", warn=True, pty=True)


@task
def mypy(c):
    """Run static type checking: mypy."""
    print(colorize("-> Running mypy..."))
    c.run(f"poetry run mypy {c.project_slug}", warn=True, pty=True)


@task(post=[black, flake, pylint, mypy], default=True)
def all(c):
    """Run all quality checks."""
