from invoke import task
from .colors import colorize
from .system import OperatingSystem, get_current_system


@task(default=True)
def update(c):
    """Update the project using cruft

    You can control the update process by modifying the .cruft.json file.
    More info on https://cruft.github.io/cruft/#updating-a-project
    """

    system = get_current_system()

    if system == OperatingSystem.LINUX:
        print_conf = "cat .cruft.json"
        update= "poetry run cruft update"
        cmd = print_conf + " && " + update
        c.run(cmd, pty=True, echo=True)
    else:
        raise ValueError(f'System {system} is not supported')
