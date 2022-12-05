#!/bin/sh -xe
pyenv update
pyenv install --skip-existing $(cat .python-version)
poetry install --no-root
