import logging
import pytest
import sys
import os
import json
from distutils import dir_util

logger = logging.getLogger(__name__)

VERSION = "1.0.14"

# Add custom options to pytest. Here we define the addition of an 'url' argument
# so you could run `pytest --url <url>` to specify an external url to check real time
# api calls for instance.
# Also it can skip tests that need this fixture but the argument was not passed to the
# pytest command.
def pytest_addoption(parser):
    """Adds option to pytest command

    '--url <url>' to specify the url of the api during CI
    """
    parser.addoption("--url", action="store")


@pytest.fixture
def url(request):
    """Returns URL if defined or skips test"""
    url_value = request.config.option.url
    if url_value is None:
        pytest.skip()
    return url_value


# Isolated data folder to tests things that require files.
@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Usage:
    - If test module is called: test_files.py
    - Create folder tests/test_files (same name as test module file)
    - Put <filename> in that folder
    - Create tests like:

        >>> def test_file_stuff(datadir):
        >>>     path = datadir.join("<filename>")
        >>>     # Open file using `path`
        >>>     # Test something with file contents
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
