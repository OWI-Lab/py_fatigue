import os
from .colors import colorize
from pathlib import Path
from invoke import task
from .system import OperatingSystem, get_current_system

@task(default=True, help={"pattern": "Pattern to search against", "extra_flags": "Extra parameters for grep"}, optional=["extra_flags"])
def search(c, pattern, extra_flags=None):
    """Search in relevant folder of the project
    """
    system = get_current_system()
    if system == OperatingSystem.LINUX:
        cmd = f"""grep -nriI . -e "{pattern}" --color=auto"""
        cmd = cmd + " --exclude-dir={notebooks/,.venv/,static,_build,.mypy_cache,htmlcov,.jupyter,.vim,.git}"
        cmd = cmd + " --exclude={poetry.lock,.pylintrc,.gitchangelog.rc,pylint_exit_handler.sh}"

        if extra_flags:
            cmd = cmd + extra_flags

        c.run(cmd, pty=True, echo=True)
    else:
        raise ValueError(f'System {system} is not supported')
