Why a template?
===============

When we think about software projects, we often think just about the resulting reports, insights, or visualizations. While these end products are generally the main event, it is easy to focus on making the products look nice and ignore the quality of the code that generates them. Because these end products are created programmatically, code quality is still important. **Ultimately, software quality is about correctness and reproducibility**.

It is no secret that a software deliverable is often the result of very scattershot and serendipitous explorations. Tentative experiments and rapidly testing approaches that might not work out are all part of the process for getting to the good stuff, and there is no general magic bullet.

That being said, once started it is not a process that lends itself to thinking carefully about the structure of your code or project layout, so it is best to start with a clean, logical structure and stick to it throughout. We believe it is a pretty big win all around to use a fairly standardized setup like this one. 

Quick 'n dirty approaches might be initially faster but over time they will become complicated to reuse, hard to maintain and eventually obsolete. Of course, recreating a standardized setup for your projects time after time is also error prone and tedious which is why this template aims to incorporate the best and practical things you will need as a Python developer alleviating the burden of setting up your dev environment.

The scope of this project template is limited to deliver a **versioned python package**. In our experience, this is the best approach to be able to focus on one thing only and do it well. Applying a modular approach in the devops cycle enables much more flexibility down the road. This template makes it easy to encapsulate your developed source code into a package which can be published on a central repository, like pypy or a private artifactory.

Project Organization
--------------------
Overview of the organization within this project. It's a high level breakdown of the different folders with
a short indication of what's the concept behind them.

::

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project
    │
    ├── references         <- Manuals, and all other explanatory materials, if needed.
    │
    ├── pyproject.toml     <- The poetry config file for reproducing the environment, e.g.
    │                         generated with `poetry install` and activated with `poetry shell`
    │
    ├── python_simple_project_1
    │   └── __init__.py    <- Makes this folder a Python package. Organize your package as you wish.
    |                         Use subpackages and modules wisely. Think about naming.
    |                         Check import logic using the notebook server.
    │
    │── tasks              <- Configuration file for defining tasks for py-invoke,
    |                         see `inv --list` or `inv -l`
    │
    └── tests              <- Tests collection for pytest

Disagree with a couple of the default folder names? Working on a project that's a little nonstandard and doesn't exactly
fit with the current structure? Prefer to use a different package than one of the defaults?

**Go for it!** This is a lightweight structure, and is intended to be a good starting point for many projects.

    Consistency within a project is more important. Consistency within one module or function is the most important.
    **However**, know when to be inconsistent! Sometimes style guide recommendations just aren't applicable.
    When in doubt, use your best judgment. Look at other examples and decide what looks best. And don't hesitate to ask!

    -- PEP8

Opinions
========
There are some opinions implicit in the project structure that have grown out of our experience with what works and
what doesn't when collaborating on python projects. Some of the opinions are about workflows, and some of the
opinions are about tools that make life easier.

Build from the environment up
-----------------------------
At any moment in time (code commit) the exact same environment can be rebuild making it easy to apply hotfixes, let others collaborate etc.
You need the same tools, the same libraries, and the same versions to make everything play nicely together.

This project template can use ``pyenv`` to setup the python environment and ``poetry`` to manage the python dependencies plus all additional tooling to help you along the way.

Keep secrets and configuration out of version control
-----------------------------------------------------
You really don't want to leak your AWS secret key or Postgres username and password on Github.
Enough said — see the Twelve Factor App principles on this point. Here's one way to do this:

Store your secrets and config variables in a special file.
Create a .env file in the project root folder. Thanks to the .gitignore, this file should never get committed into the version control repository. Here's an example:

::

    # example .env file
    DATABASE_URL=postgres://username:password@localhost:5432/dbname
    AWS_ACCESS_KEY=myaccesskey
    AWS_SECRET_ACCESS_KEY=mysecretkey
    OTHER_VARIABLE=something

Use a package to load these variables automatically like `python-dotenv`.

This is not explicitely added to this template. Just remember to NEVER add secrets to version control. Even if it's just temporary.

Be conservative in changing the default folder structure
--------------------------------------------------------
To keep this structure broadly applicable for many different kinds of projects, we think the best approach is to be liberal
in changing the folders around for your project, but be conservative in changing the default structure for all projects.

Result is a package
===================
The scope of this project template is limited to deliver a **versioned python package**. In our opinion it's best
to focus on one thing only and do it well. Applying a modular approach in the devops cycle enables much more flexibility down the road.
This template makes it easy to encapsulate your developed source code into a package which can be published on a central repository,
like pypy or a private artifactory if you have access to one. Even if you only build locally or as an artifact attached to
the git repository you can easily integrate it somewhere else. For instance, other projects where you want to reuse your
functionality, notebook environments, but it also makes it straightforward to expose your functionality in a dedicated API template or
containerized application.

Separation of concerns
----------------------
A common workflow for deploying functionality into production, like an API, can be the following:

Create and publish a python package -> Import package, expose (subset) of functionality via endpoint and isolate it in a container image -> fetch image and deploy. 

It's clear that each of those steps have their own best practices on development, testing and documentation so it makes a lot of sense to split them in dedicated projects. Shifting from a quick and dirty approach, people tend to chain these different projects locally. It allows them to change something in one project and see the result directly. While this can be a valid approach, and it's a lot better then a monolithic approach, it opens the door for a quick and dirty way of working. Tests are often bypassed, documentation discarded because only the end result is important. 

What if you would take another step back and look at it from an isolated point of view. You develop a Python package with certain functionality, you as a developer take care of testing and documenting the functionality and make sure everything checks out before you release and publish a new version. You can focus on the functionality aspect alone and don't have to care about how it will be used. It's not your concern. 

Yourself or someone else can now reuse that functionality easily and focus on how to create an interface for it. This can take a lot of forms, think about jupyter notebooks, APIs, CLIs,.. The only thing you need to know is how to read the manual of the package so you import and use the classes and functions. 

Let's follow our example and expose a subset of the package via an api endpoint with specific parameters. You import a function from the package in a separate package and focus on writing a parameterized endpoint with validation, meaningfull http codes etc. You add functional tests and isolate the api & webserver in a dedicated and optimized container image. You can now run this container wherever you want, it can be on kubernetes, via docker compose,.. it's not your concern.

If you tend to follow these approach and adhere to the separation of concerns paradigm you can truly focus on each aspect separately. You can split the work between each other as the level of expertise needed to work on an application is substantially decreased. Especially when you start from dedicated templates which are carefully crafted to follow best-practices in particular development cases. 

Quite often you hear that the latency of building all these artifacts via separate commits and PRs is too high before you can check something in real life (deployed). In theory they are still following a quick and dirty route without putting much thought in what is actually needed to accomplish a feature. When you use an agile devops methodology is perfectly feasible to split a feature into the different steps: code, expose and deploy. 

