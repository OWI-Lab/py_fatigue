import shutil
import os

from invoke import task
from .colors import colorize
from .system import get_current_system, OperatingSystem, DOCS_BUILD_DIR, DOC_SCREEN_NAME



@task
def clean(c):
    """Clean documentation build folder"""
    if os.path.exists(DOCS_BUILD_DIR):
        shutil.rmtree(DOCS_BUILD_DIR)


@task
def build(c):
    """Run sphinx to build documentation."""
    system = get_current_system()

    if system == OperatingSystem.LINUX:
        _command = f"poetry run sphinx-build -b  html -d {DOCS_BUILD_DIR}/doctrees ./rst_docs {DOCS_BUILD_DIR}/html -E -a"
    elif system == OperatingSystem.WINDOWS:
        _command = f"poetry run sphinx-build -b html -d {DOCS_BUILD_DIR}\\doctrees rst_docs {DOCS_BUILD_DIR}\\html -E -a"
    else:
        raise ValueError(f'System {system} is not supported')

    c.run(_command)


@task
def start(c):
    """Start documentation webserver."""
    SERVER_PORT = int(c.start_port) + 1
    system = get_current_system()

    if system == OperatingSystem.LINUX:
        _command = f"poetry run screen -d -S {DOC_SCREEN_NAME} \
            -m python -m http.server --bind localhost \
            --directory {DOCS_BUILD_DIR}/html {SERVER_PORT}"
    elif system == OperatingSystem.WINDOWS:
        _command = f"poetry run wt -d . " \
                   f"python -m http.server --bind localhost " \
                   f"--directory {DOCS_BUILD_DIR}\\html {SERVER_PORT} "
    else:
        raise ValueError(f'System {system} is not supported')

    c.run(_command)

    url = f"http://localhost:{SERVER_PORT}"

    print("Documentation hosted in background:\n")
    print(f"-> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv docs.stop')}\n")


@task
def stop(c):
    """Stop documentation webserver."""
    system = get_current_system()

    if system == OperatingSystem.LINUX:
        result = c.run(f"screen -ls {DOC_SCREEN_NAME}", warn=True, hide="both")
        if "No Sockets" in result.stdout:
            return
        screens = result.stdout.splitlines()[1:-1]
        for s in screens:
            name = s.split("\t")[1]
            c.run(f"screen -S {name} -X quit")
    elif system == OperatingSystem.WINDOWS:
        print("Documentation server is not attached to this process. Close windows terminal instance instead")
        return
    else:
        raise ValueError(f'System {system} is not supported')


@task
def linkcheck(c):
    """Check external links in documentation."""
    system = get_current_system()

    if system == OperatingSystem.LINUX:
        _command = f"poetry run sphinx-build -b linkcheck -d {DOCS_BUILD_DIR}/doctrees " \
                   f"./rst_docs {DOCS_BUILD_DIR}/linkcheck"
    elif system == OperatingSystem.WINDOWS:
        _command = f"poetry run sphinx-build -b linkcheck -d {DOCS_BUILD_DIR}\\doctrees " \
                   f"rst_docs {DOCS_BUILD_DIR}\\linkcheck"
    else:
        raise ValueError(f'System {system} is not supported')

    c.run(
        _command,
        warn=True,
        pty=True,
    )


@task(post=[stop, clean, build, start], default=True)
def rebuild(c):
    """Rebuild documentation and start documentation webserver."""
