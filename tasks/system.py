import platform
from enum import Enum


class OperatingSystem(Enum):
    WINDOWS = 'Windows'
    LINUX = 'Linux'
    MAC = 'Darwin'


def get_current_system():
    system = platform.system()

    if system == 'Linux':
        return OperatingSystem.LINUX
    if system == 'Windows':
        return OperatingSystem.WINDOWS
    if system == 'Darwin':
        return OperatingSystem.MAC

    raise ValueError(f'Invalid operating system: {system}')


system = get_current_system()

if system == OperatingSystem.LINUX:
    COV_SCREEN_NAME = "coverage"
    COV_DOC_BUILD_DIR = "_build/htmlcov"
    DOCS_BUILD_DIR = "_build/docs"
    DOC_SCREEN_NAME = "sphinx-docs"
elif system == OperatingSystem.WINDOWS:
    COV_DOC_BUILD_DIR = r"_build\htmlcov"
    DOCS_BUILD_DIR = r"_build\docs"
else:
    raise ValueError(f'System {system} is not supported')
