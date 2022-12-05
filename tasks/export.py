from invoke import task
from .colors import colorize

WHEEL_DIR = 'export/wheels'


@task
def collect(c):
    """Collect all dependencies wheels."""

    print(f"\n-> {colorize('Generate wheels...')}\n")

    c.run(f'pip wheel . --wheel-dir {WHEEL_DIR}', echo=True)

@task
def deptree(c):
    """Generate dependency tree."""

    print(f"\n-> {colorize('Dependency tree...')}\n")

    c.run(f'poetry show -t -l --no-dev', pty=True, echo=True)

@task
def archive(c):
    """Archive exports"""

    print(f"\n-> {colorize('Archive...')}\n")

    # Make symbolic link
    project_version = c.run("sed 's/ /-/g' <<< $(poetry version)", echo=True).stdout.strip()
    c.run(f'ln -s ./export ./export-{project_version}', echo=True)

    # Zip folder, ignore warnings
    c.run(f'zip -r export-{project_version}.zip ./export-{project_version}/', echo=True, warn=True)

    # Remove symlink
    c.run(f'rm export-{project_version}', echo=True)

@task(post=[deptree, collect, archive], default=True)
def export(c):
    """Collect wheels & archive."""