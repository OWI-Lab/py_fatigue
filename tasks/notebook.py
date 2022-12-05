import glob
from invoke import task
from .colors import Color, colorize

NB_PORT = 9000
SCREEN_NAME = 'notebook'

@task()
def run(c):
    """Start notebook server on foreground."""

    with c.cd("./notebooks/"):
        with c.prefix("export JUPYTER_CONFIG_DIR=../.jupyter"):
            c.run(f"poetry run jupyter notebook --port={NB_PORT} --no-browser")


@task
def stop(c):
    """Stop notebook server in background."""
    result = c.run(f'screen -ls {SCREEN_NAME}', warn=True, hide='both')
    if 'No Sockets' in result.stdout:
        return
    screens = result.stdout.splitlines()[1:-1]
    for s in screens:
        name = s.split('\t')[1]
        c.run(f"screen -S {name} -X quit")


@task(pre=[stop], default=True)
def start(c):
    """Start notebook server in background."""

    with c.cd("./notebooks/"):
        with c.prefix("export JUPYTER_CONFIG_DIR=../.jupyter"):
            c.run(f"poetry run screen -d -S {SCREEN_NAME} -m jupyter notebook  --port={NB_PORT} --no-browser")

    url = f'http://localhost:{NB_PORT}'

    print(f"Jupyter hosted in background:\n")
    print(f"-> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv nb.stop')}\n")


@task
def clean(c, check=False):
    """Clean all notebooks in folder."""
    notebook_files = glob.glob("./notebooks/*.ipynb")
    command = ("clean", "Cleaning")
    if check:
        command = ("check", "Checking")

    for f in notebook_files:
        print(f"{command[1]} {f}")
        CMD = f'nb-clean {command[0]} -e {f}'
        print(f"-> {CMD}")
        c.run(CMD)