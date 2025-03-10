import glob
from invoke import task
from .colors import colorize, Color
from .system import (
    OperatingSystem,
    get_current_system,
)

SCREEN_NAME = "notebook"
SYSTEM = get_current_system()


@task
def run(c_r):
    """Start notebook server on foreground."""
    with c_r.cd("./notebooks"):
        # check current directory
        with c_r.prefix("export JUPYTER_CONFIG_DIR=../.jupyter"):
            _command = f"jupyter notebook --port={c_r.start_port} --no-browser --NotebookApp.token='' --NotebookApp.password=''"
            c_r.run(_command)


@task
def stop(c_r):
    """Stop notebook server in background."""
    result = c_r.run(f"screen -ls {SCREEN_NAME}", warn=True, hide="both")
    if "No Sockets" in result.stdout:
        return
    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        tmp_str = colorize(
            "Stopping notebook server...",
            color=Color.HEADER,
            bold=True
        )
        print(f"{tmp_str}")
        _command = f"kill $(lsof -ti:{c_r.start_port})"
        print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
        c_r.run(_command)
    elif SYSTEM == OperatingSystem.WINDOWS:
        print(
            "Stopping notebook server is not supported on Windows. "
            "Please stop the server manually."
        )
    else:
        raise ValueError(f"System {SYSTEM} is not supported")


@task(pre=[stop], default=True)
def start(c_r):
    """Start notebook server in background."""

    tmp_str = colorize(
        "Starting notebook server...",
        color=Color.HEADER,
        bold=True
    )
    with c_r.cd("./notebooks"):
        if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
            with c_r.prefix("export JUPYTER_CONFIG_DIR=../.jupyter"):
                _command = (
                    f"screen -d -S {SCREEN_NAME} -m "
                    f"jupyter notebook --port={c_r.start_port} --no-browser "
                    "--NotebookApp.token='' --NotebookApp.password=''"
                )
                print(f"{tmp_str}")
                c_r.run(_command)
                print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
        elif SYSTEM == OperatingSystem.WINDOWS:
            with c_r.cd("./notebooks"):
                _command = (
                    "wt -d . jupyter notebook "
                    f"--port={c_r.start_port} --no-browser --NotebookApp.token='' --NotebookApp.password=''"
                )
                print(
                    colorize(
                        "Notebook server is not attached to this terminal "
                        " process. Close windows terminal instance instead.",
                        color=Color.WARNING,
                    )
                )

            print(f"{tmp_str}")
            c_r.run(_command)
            print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
        else:
            raise ValueError(f"System {SYSTEM} is not supported")

    url = f'http://localhost:{c_r.start_port}'

    print("Jupyter hosted in background:\n")
    print(f"--> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv nb.stop')}\n")


@task
def clean(c_r, check=False):
    """Clean all tutorials in folder."""
    notebook_files = glob.glob("./notebooks/*.ipynb")
    command = ("clean", "Cleaning")
    if check:
        command = ("check", "Checking")

    for nb_file in notebook_files:
        print(f"{command[1]} {nb_file}")
        _command = f"nb-clean {command[0]} -e {nb_file}"
        print(f"--> {_command}")
        c_r.run(_command)
