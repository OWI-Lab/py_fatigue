from invoke import task
from .colors import colorize
from .system import (
    OperatingSystem,
    get_current_system,
    COV_DOC_BUILD_DIR,
    COV_SCREEN_NAME,
)


@task(help={"verbose": "Run tests verbose."})
def run(c, verbose=False):
    """Run test suite."""
    # print('I am run(c) and I was called!')
    # print('I am c!\n', c, '\n', type(c), '\n', c.__dict__)  # <-- Who's c?
    if verbose:
        c.run(
            f"poetry run pytest -v -W ignore::UserWarning \
            --cov={c.project_slug} --cov-report=term:skip-covered \
            --cov-report=html --cov-report=html:{COV_DOC_BUILD_DIR}",
            pty=True,
        )
    else:
        c.run(
            f"poetry run pytest -W ignore::UserWarning \
            --cov={c.project_slug} --cov-report=term:skip-covered \
            --cov-report=html --cov-report=html:{COV_DOC_BUILD_DIR}",
            pty=True,
        )


@task
def coverage(c):
    """Start coverage report webserver."""
    COV_PORT = c.start_port + 2
    system = get_current_system()

    if system == OperatingSystem.LINUX:
        _command = f"poetry run screen -d -S {COV_SCREEN_NAME} \
            -m python -m http.server --bind localhost \
            --directory {COV_DOC_BUILD_DIR} {COV_PORT}"
    elif system == OperatingSystem.WINDOWS:
        _command = f"wt -d . python -m http.server --bind localhost --directory {COV_DOC_BUILD_DIR} {COV_PORT}"
    else:
        raise ValueError(f"System {system} is not supported")

    c.run(_command)

    url = f"http://localhost:{COV_PORT}"

    print("Coverage server hosted in background:\n")
    print(f"-> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv test.stop')}\n")


@task
def stop(c):
    """Stop coverage report webserver."""
    system = get_current_system()

    if system == OperatingSystem.LINUX:
        result = c.run(f"screen -ls {COV_SCREEN_NAME}", warn=True, hide="both")
        if "No Sockets" in result.stdout:
            return
        screens = result.stdout.splitlines()[1:-1]
        for s in screens:
            name = s.split("\t")[1]
            c.run(f"screen -S {name} -X quit")
    elif system == OperatingSystem.WINDOWS:
        print(
            "Coverage server is not attached to this process. Close windows terminal instance instead"
        )
        return
    else:
        raise ValueError(f"System {system} is not supported")


@task(post=[stop, run, coverage], default=True)
def all(c):
    """Run all tests and start coverage report webserver."""
    # print('I am all(c) and I was called!')