Project tools
=============
This page guides you through the different tools which are available in this project template.
For each tool we limit the documentation to their main usage so you can start using them.
When relevant, external links are provided to the original documentation of each tool. You are
encouraged to read through it to get a deeper understanding.

pyenv
-----
lets you easily **switch between multiple versions of Python**. It's simple, unobtrusive,
and follows the UNIX tradition of single-purpose tools that do one thing well.

It's the prerequisite for the template and should already be installed on your system. A good
introduction can be found on `Real Python <https://realpython.com/intro-to-pyenv/>`_.

The most relevant command is ``pyenv versions`` which displays all currently installed
Python versions on your system.

screen
------
is a full-screen window manager that multiplexes a physical terminal between several
processes (typically interactive shells). It's used within this project to start local web servers
in the background.

An overview of the command line options can be found
`here <https://www.gnu.org/software/screen/manual/screen.html#Invoking-Screen>`_.
Normally you will not directly need this tool but ``screen -ls`` is handy to check all
running background tasks and ``killall screen`` when you need to close all at once.

.. NOTE::
   In case a **Permission denied** error pops up you can fix it with running the next commands
   in a terminal:
 
   .. code-block:: bash

      mkdir ~/.screen && chmod 700 ~/.screen
      export SCREENDIR=$HOME/.screen

   Additionally you can put the last line into you `~/.bashrc` (or alternative shell like .zshrc)
   so that it will also take effect afterwards.

poetry
------
is a tool for **dependency management and packaging** in Python.
It allows you to declare the libraries your project depends on and it will manage (install/update) them for you.

Since you are reading this documentation this tool should already be installed with the ``exec-helper create_environment``
command. Be sure to read the `basic usage <https://python-poetry.org/docs/basic-usage/>`_ and
`commands <https://python-poetry.org/docs/cli/>`_ documentation on the poetry website.

Relevant ``poetry <command>`` you will need within the project:

:shell:     Spawns a shell within the virtual environment.
            If one doesn't exist yet, it will be created.

:add:       Adds required packages to your `pyproject.toml` and installs them.

            -D, --dev      Add package as development dependency.
            --path         The path to a dependency.

:remove:    Removes a package from the current list of installed packages.

            -D, --dev      Removes a package from the development dependencies.

:show:      List all of the available packages.

            --no-dev          Do not list the dev dependencies.
            --tree            List the dependencies as a tree.
            -l, --latest      Show the latest version.
            -o, --outdated    Show the latest version but only for packages that are outdated.

:build:     Builds the source and wheels archives.

            -F format, --format=format  Limit the format to either wheel or sdist.

:update:    In order to get the latest versions of the dependencies and to update the `poetry.lock` file.
            You should use this command to update all packages to the latest versions.

            --no-dev          Do not install dev dependencies.

During initialization of the project when running the ``exec-helper create_environment`` command, ``poetry install``
was launched in the background. Since there was no `poetry.lock` file poetry resolved all dependencies in
the `pyproject.toml` file and installed them.
When Poetry has finished installing, it writes all of the packages and the exact versions of them that it
downloaded to the `poetry.lock` file, locking the project to those specific versions.
You should commit the `poetry.lock` file to your project repo so that all people working on the project
are locked to the same versions of dependencies.

Running install when a poetry.lock file is present resolves and installs all dependencies that you
listed in `pyproject.toml`, but Poetry uses the exact versions listed in `poetry.lock` to ensure that the
package versions are consistent for everyone working on your project. As a result you will have all dependencies
requested by your `pyproject.toml` file, but they may not all be at the very latest available versions.
This is by design, it ensures that your project does not break because of unexpected changes in dependencies.

To update to the latest versions, use the ``update`` command. This will fetch the latest matching versions
 (according to your `pyproject.toml` file) and update the lock file with the new versions.

black
-----
is the uncompromising **automated Python code formatter**. Black gives you speed, determinism,
and freedom from PyCodeStyle nagging about formatting. You will save time and mental
energy for more important matters. In other words it auto-corrects common formatting mistakes
and makes code reviews faster by producing the smallest diffs.
See `black documentation <https://github.com/psf/black>`_ to learn more.

It reformats entire files in place. It is not configurable. It doesn't take previous formatting into account.
It doesn't reformat blocks that start with ``# fmt: off`` and end with ``# fmt: on``.

flake8
------
is a command-line utility for **enforcing style consistency** across Python projects. By default it includes
lint checks provided by the PyFlakes project, PEP-0008 inspired style checks provided by the PyCodeStyle project,
and McCabe complexity checking provided by the McCabe project.
More information can be found on the flake8 `docs <https://flake8.pycqa.org/en/latest/index.html>`_.

pylint
------
is a tool that checks for errors in Python code, tries to enforce a coding standard and looks for code smells.
It can also look for certain type errors, it can recommend suggestions about how particular blocks can be
refactored and can offer you details about the code's complexity.

Within this template it's setup to **provide suggestions**. Check out the `manual <http://pylint.pycqa.org/en/latest/>`_
if you want to learn more.

If you can't live with all the suggestion about you can limit the output by adding the option ``--disable=R,C``
to the pylint command.

mypy
----
is an optional **static type checker** for Python that aims to combine the benefits of dynamic (or "duck") typing
and static typing. Mypy combines the expressive power and convenience of Python with a powerful type system
and compile-time type checking.

Using the Python 3 function annotation syntax (using the PEP 484 notation), you will be able to efficiently
annotate your code and use mypy to check the code for common errors.
Mypy has a powerful and easy-to-use type system with modern features such as type inference,
generics, callable types, tuple types, union types, and structural sub-typing.

Check out `mypy docs <https://mypy.readthedocs.io/en/stable/>`_ and the
`getting started guide <https://mypy.readthedocs.io/en/stable/getting_started.html>`_.

.. NOTE::
   Black, flake8, pylint and mypy are bundled under the **quality assurance** command. Run ``eh qa`` to run
   them all sequentially.

sphinx
------
is a tool that translates a set of reStructuredText source files into various output formats,
automatically producing cross-references, indices, etc. If you have a directory containing a
bunch of reST-formatted documents (and possibly subdirectories of docs in there as well),
Sphinx can generate a nicely-organized arrangement of HTML files (in some other directory) for easy browsing
and navigation. But from the same source, it can also generate a PDF file using LaTeX.

Sphinx is used within this project to **generate the documentation** you are reading right now.
Be sure to check the `reStructuredText guide <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
to understand the syntax. Compared to markdown it offers quite a bit more functionality and it's
not that much more difficult.
Also check the `Sphinx Getting Started <https://www.sphinx-doc.org/en/master/usage/quickstart.html#autodoc>`_ guide
when you need to know more advanced use cases.

When documenting Python code, it is common to put a lot of documentation in the source files, in documentation strings.
Sphinx supports the inclusion of docstrings from your modules with autodoc whcih can be used to convert your source
documentation in a nice looking API.

This project uses the `napoleon sphinx extension <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_
which interprets every docstring that Sphinx **autodoc** can find, including docstrings on: modules, classes,
attributes, methods, functions, and variables.
Inside each docstring, specially formatted `Sections` are parsed and converted to reStructuredText.

The napoleon extension supports two styles of docstrings: Google and NumPy.
We think the google style is the cleanest, check the
`example <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google>`_ for
yourself, so we used that one to generate the `version module` documentation.

.. NOTE::
   Use PEP 484 type annotations!

   PEP 484 introduced a standard way to express types in Python code. This is an alternative
   to expressing types directly in docstrings. One benefit of expressing types according to
   PEP 484 is that type checkers and IDEs can take advantage of them for static code analysis.
   Since **mypy** is part of the project as well the benefit is twofold.

.. NOTE::
   To check if all your external links are still valid references in your documentation you should
   run ``eh linkcheck`` from time to time.

pytest
------
The pytest framework makes it easy to **write small tests, yet scales to support complex functional testing** for applications and libraries.
Within this project you can run pytest to look for tests in the `tests/` directory. You can run it directly with the ``pytest`` command or
take advantage of the pre-defined task (``eh test``).

The functionality of this test suite is extensive and there are a bunch of plugins written for it. Be sure to browse
the documentation at https://docs.pytest.org/en/latest/contents.html#toc.
It's also easy to extend it with your own custom functionality. In the `tests/conftest.py` there are some examples of this.

bumpversion
-----------
Version-bump your software with a single command! A small command line tool to simplify releasing
software by updating all version strings in your source code by the correct increment.

Within this project this tool is used to keep the project version in sync between different files.

Use the ``bumpversion major|minor|patch`` command to update the required part of the semver version tag.

gitchangelog
------------
Use your commit log to make beautiful and configurable changelog file.
See https://pypi.org/project/gitchangelog/ to learn more.

In practice if you follow these basic commit message rules you can generate a nice
changelog without any additional work. So why not be more disciplined when writing
commit messages? The overhead is minimal, others and the future you will thank you!

.. code-block:: bash

   ACTION: [AUDIENCE:] COMMIT_MSG [!TAG ...]

   Description

   ACTION is one of 'chg', 'fix', 'new'

         Is WHAT the change is about.

         'chg' is for refactor, small improvement, cosmetic changes...
         'fix' is for bug fixes
         'new' is for new features, big improvement

   AUDIENCE is optional and one of 'dev', 'usr', 'pkg', 'test', 'doc'

         Is WHO is concerned by the change.

         'dev'  is for developpers (API changes, refactors...)
         'usr'  is for final users (UI changes)
         'pkg'  is for packagers   (packaging changes)
         'test' is for testers     (test only related changes)
         'doc'  is for doc guys    (doc only changes)

   COMMIT_MSG is ... well ... the commit message itself.

   TAGs are additionnal adjective as 'refactor' 'minor' 'cosmetic'

         They are preceded with a '!' or a '@' (prefer the former, as the
         latter is wrongly interpreted in github.) Commonly used tags are:

         'refactor' is obviously for refactoring code only
         'minor' is for a very meaningless change (a typo, adding a comment)
         'cosmetic' is for cosmetic driven change (re-indentation, 80-col...)
         'wip' is for partial functionality but complete subfunctionality.

   Example:

   new: usr: support of bazaar implemented
   chg: re-indentend some lines !cosmetic
   new: dev: updated code to be compatible with last version of killer lib.
   fix: pkg: updated year of licence coverage.
   new: test: added a bunch of test around user usability of feature X.
   fix: typo in spelling my name in comment. !minor

   Please note that multi-line commit message are supported, and only the
   first line will be considered as the "summary" of the commit message. So
   tags, and other rules only applies to the summary.  The body of the commit
   message will be displayed in the changelog without reformatting.


Jupyter notebook
----------------
Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.

This project has built-in functionality to launch a jupyter notebook environment, linked to your Python environment. Its `autoreload` magic creates a live link between this project's source folder and the running web notebook.

To start your local jupyter notebook server run ``inv nb.start`` and open the **`0.0-ddw-nb-development-guide.ipynb`** notebook for a development guide on how to get started with this approach.

The jupyter project is also enormous but dedicated documentation for the notebook can be found `in the official documentation <https://jupyter-notebook.readthedocs.io/en/stable/>`_.

