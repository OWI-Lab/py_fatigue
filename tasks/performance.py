from invoke import task
from .system import OperatingSystem, get_current_system


@task(default=True)
def profile(c):
    """Create performance profile and show it in timeline"""
    system = get_current_system()
    if system == OperatingSystem.LINUX:
        cmd = fr"""pyinstrument {c.project_slug}/__init__.py | grep "pyinstrument --load-prev" | sed 's/\[options\]/-r html/' | source /dev/stdin -f"""
        c.run(cmd, pty=True)
    else:
        raise ValueError(f'System {system} is not supported')
