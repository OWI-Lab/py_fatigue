import shutil
import os

from invoke import task

from .colors import colorize, Color
from .system import (
    OperatingSystem,
    get_current_system,
    DOCS_BUILD_DIR,
    DOC_SCREEN_NAME,
    PTY,
)

SYSTEM = get_current_system()

@task
def clean(c_r):
    """Clean documentation build folder"""
    if os.path.exists(DOCS_BUILD_DIR):
        shutil.rmtree(DOCS_BUILD_DIR)


@task
def autobuild(c_r):
    """Run sphinx to build documentation."""
    DOCS_PORT = c_r.start_port + 1

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        _command = f"sphinx-autobuild -b html -d {DOCS_BUILD_DIR}/doctrees ./rst_docs {DOCS_BUILD_DIR} -E -a --port {DOCS_PORT} > /dev/null"
    elif SYSTEM == OperatingSystem.WINDOWS:
        _command = f"sphinx-autobuild -b html -d {DOCS_BUILD_DIR}\\doctrees rst_docs {DOCS_BUILD_DIR} -E -a --port {DOCS_PORT} > NUL"
    else:
        raise ValueError(f'System {SYSTEM} is not supported')
    tmp_str = colorize(
        "Starting documentation server with hot reload...\n",
        color=Color.HEADER,
        bold=True
    )
    print(tmp_str)

    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")

    url = f"http://localhost:{DOCS_PORT}"

    print("Documentation server with hot reload hosted at:\n")
    print(f"--> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv docs.stop')}\n")
    tmp_str = colorize(
        "Waiting for documentation server to be ready...",
        color=Color.HEADER,
        bold=True
    )
    print(tmp_str, end="\r")

    c_r.run(_command, pty=PTY)


@task
def serve(c_r):
    """Start documentation webserver."""
    DOCS_PORT = c_r.start_port + 1

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        _command = f"screen -d -S {DOC_SCREEN_NAME} " \
                   f"-m python -m http.server --bind localhost " \
                   f"--directory {DOCS_BUILD_DIR} {DOCS_PORT}"
    elif SYSTEM == OperatingSystem.WINDOWS:
        _command = f"wt -d . " \
                   f"python -m http.server --bind localhost " \
                   f"--directory {DOCS_BUILD_DIR} {DOCS_PORT}"
    else:
        raise ValueError(f'System {SYSTEM} is not supported')
    tmp_str = colorize(
        "Starting documentation server...",
        color=Color.HEADER,
        bold=True
    )
    print(tmp_str)
    c_r.run(_command)
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")

    url = f"http://localhost:{DOCS_PORT}"

    print("Documentation server hosted in background:\n")
    print(f"--> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv docs.stop')}\n")


@task
def stop(c_r):
    """Stop documentation webserver."""
    DOCS_PORT = c_r.start_port + 1

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        result = c_r.run(
            f"screen -ls {DOC_SCREEN_NAME}", warn=True, hide="both"
        )
        if "No Sockets" in result.stdout:
            return
        tmp_str = colorize(
            "Stopping coverage server...",
            color=Color.HEADER,
            bold=True
        )
        print(tmp_str)
        _command = f"kill $(lsof -ti:{DOCS_PORT})"
        print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
        c_r.run(_command)

    elif SYSTEM == OperatingSystem.WINDOWS:
        print(
            colorize(
                "Coverage server is not attached to this terminal process. "
                "Close windows terminal instance instead.",
                color=Color.WARNING,
            )
        )
        return
    else:
        raise ValueError(f"System {SYSTEM} is not supported")


@task
def linkcheck(c_r):
    """Check external links in documentation."""
    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        _command = f"sphinx-build -b linkcheck -d {DOCS_BUILD_DIR}/doctrees " \
                   f"./rst_docs {DOCS_BUILD_DIR}/linkcheck"
    elif SYSTEM == OperatingSystem.WINDOWS:
        _command = f"sphinx-build -b linkcheck -d {DOCS_BUILD_DIR}\\doctrees " \
                   f"rst_docs {DOCS_BUILD_DIR}\\linkcheck"
    else:
        raise ValueError(f'System {SYSTEM} is not supported')

    c_r.run(_command, warn=True, pty=PTY)


@task(post=[stop, clean, autobuild], default=True)
def rebuild(c_r):
    """Rebuild documentation and start documentation webserver."""
