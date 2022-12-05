# Getting started
## Supported systems

This template supports:
- **NIX** systems (tested on: Ubuntu)
- WSL on Windows (no native Windows currently)
- Native Windows 10, with Windows Terminal installed

## Prerequisites

## The simple way - devcontainer
When using Visual Studio Code you can take advantage of the [ms-vscode-remote.remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension to work inside a development container. The `.devcontainer` definition holds all prerequisites to work on a project.

So what do you need? 

1. A docker-dektop installation with WSL2 backend. Check the [guide](https://docs.docker.com/desktop/windows/wsl/) on how to install.
2. Visual Studio Code with the *Remote - Containers* extension. See [here](https://code.visualstudio.com/docs/remote/containers)
3. [Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/installation.html).

Now when you generate the project, VSC will ask to launch the project in a development container. Go ahead, the container image will be build and the VSC will reopen in the container remote. 

What's left is `poetry install --no-root` to get your python virtual environment installed. Now activate with `poetry shell` and you can start. Run `inv docs` for the full documentation.

## Local system

As a first step, make sure all prerequisites are properly fulfilled:

- [pyenv](https://github.com/pyenv/pyenv): A tool for managing different python versions.
    The best way to install pyenv and friends ([link](https://github.com/pyenv/pyenv-installer)) is

      > curl https://pyenv.run | bash

    For more detailed information on how to manage python versions see the [Real Python guide](https://realpython.com/intro-to-pyenv/#specifying-your-python-version>). 
    
    :::{Note}
    This project will work without `pyenv` but you loose quite a bit of functionality. Make sure your system python corresponds with the one in `pyproject.toml`.
    :::

- [poetry](https://python-poetry.org): A tool for dependency management. Check its [documentation](https://python-poetry.org/docs/) for more information. To install

      > curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

- [exec-helper](https://exec-helper.readthedocs.io/en/master/) (>0.5.1): A CLI meta-wrapper for automating common tasks. Check its [installation instructions](https://exec-helper.readthedocs.io/en/master/INSTALL.html) for installing it. This tool is optional but can help in automating some initial tasks when no python environment is available yet.

- [screen](https://www.gnu.org/software/screen/manual/screen.html) (install via `apt get install screen`) is a full-screen window manager that multiplexes a physical terminal between several 
processes (typically interactive shells). It's used within this project to start local web servers in the background.  
An overview of the command line options can be found [here](https://www.gnu.org/software/screen/manual/screen.html#Invoking-Screen>).
Normally you will not directly need this tool but `screen -ls` is handy to check all
running background tasks and `killall screen` when you need to close all at once.

    :::{Note}
    In case a **Permission denied** error pops up you can fix it with running the next commands 
    in a terminal: 

        > mkdir ~/.screen && chmod 700 ~/.screen
        > export SCREENDIR=$HOME/.screen

    Additionally you can put the last line into you `~/.bashrc` (or alternative shell like .zshrc) 
    so that it will also take effect afterwards.
    :::

## Setting up the project environment
### Bootstrap
Run from the project's root directory::

    > ./scripts/bootstrap.sh     # On Nix systems
or 
    > ./scripts/bootstrap.bat    # On Windows

This will install the chosen Python version and development dependencies as defined in your `pyproject.toml`.

### Manual install of dedicated Python version via pyenv
In case you don't have the required python version yet and you have `pyenv` installed: 

    # Will install python=<value inside .python-version>
    > eh python

    > pyenv update
    > pyenv install $(cat .python-version) # or specific version

and the command will leverage pyenv to install that specific python version.

### Manual install of Python development dependencies
After all mandatory prerequisites have been installed, you can setup a complete virtual environment using (execute this in the root or a subdirectory of this project):

    # Will install project dependencies based on `pyproject.toml`
    > eh venv 

    # Without exec-helper
    > poetry install --no-root
    > poetry shell

You can activate this environment by executing `poetry shell` or execute something in this environment using `poetry run <command>`.

## Frequent tasks automation
There are a bunch of frequently used tasks encoded within this project. They are defined in the `./tasks` folder and can be extended. Make sure the virtual environment is activated and run

    > invoke --list
    # or
    > inv -l

to get an overview of all preconfigured frequently used tasks.

## Generate documentation

This readme is only the **_tip of the iceberg_**.

To generate the additional documentation we will already leverage the tooling & task automation inside this project. 

If you bootstrapped and activated your environment you can now run:

    > inv docs

and open the given URL in your browser.

Make sure you read through the  **Development** section of the documentation to understand the why, get an overview of the features and understand how to get started.
It is strongly encouraged to take your time for this, experiment with the various invoke commands and explore the online references.
