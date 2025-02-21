.. _version-control:

Setup version control
=====================

Getting started
---------------

Every development project should be under **revision control**.
Therefore the first thing we need to do is to initialize our git repository, create a remote and push everything to `main`.

1. Initialize git inside your local project folder by running `git init`.
2. Create a remote repository. For instance, on Azure Devops, you can create (or request the project admin) a new repository within the Azure Devops project. Make sure it is a blank repository (no need for anything added).
3. Setup SSH connectivivity by adding your public ssh key to your account.
4. Push an existing repository (your local git repo created in step 1). You can do so with the following command: `git remote add origin git@ssh.dev.azure.com:v3/<Your DevOps Org/<DevOps Project>/<repository>`. The link is often given on the overview page. Your local repository now knows about the remote one and it is named `origin` by default.
5. Add all current files and push to the `main` branch:
   
   .. code-block:: bash

     # Create a local main branch
     > git checkout -b main

     # Add everything & commit
     > git add .
     > git commit -m 'New: Usr: Initialized repository [skip ci]'

     # Push to remote
     > git push --set-upstream origin main

Using just an (unprotected) main branch might be feasible when very frequent commits are combined with a strict CI/CD.
In real life, however, it's a lot easier to just **NEVER** push directly to main.

You should break apart your work in small pieces that can be done in a very short time, create a **feature/*** branch for it, commit regularly to that branch and do a merge request to integrate your functionality.

It's not uncommon for a branch to live for less then a day and to receive commits every 15-30 minutes. The goal is to keep main as up to date as possible and prevent difficult merges down the line which might occur if your branches live too long.

Luckily the templates also includes functionality to make the process of continuous delivery a lot easier and takes care of the versioning for you.

Pipelines & artifacts
---------------------

The project has continuous integration and delivery built-in. With delivery we mean that an artifact is generated and published into an artifactory where it can be consumed by other projects.

Ideally, a pipeline runs on every commit. For resource-heavy pipelines, one can consider only running on:

- `main` branch
- `releases/*` branches
- `dev` branch
- Pull requests

Create pipeline
~~~~~~~~~~~~~~~

The generated project already has a simple, fully functional pipeline for your chosen CI/CD framework (Azure DevOps, Gitlab...). You can start extending it with project-specific jobs right away!

Configure (or create) artifact feed
-----------------------------------

Depending on the CI/CD framework, you will need to configure an artifactory (a place where so-called artifacts, like build packages, are stored).

Semantic versioning
-------------------

Everything is now configured so that every push will trigger a new pipeline which will generate and **publish** a new package to an artifact feed.

The published Python package receives a semantic version tag automatically. This template uses the `Gitversion` tool to take care of that. You can read the `documentation <https://gitversion.net/docs>`_, but we will explain a sane default approach that should be sufficient for most projects.

.. note:: Install the GitVersion tool to determine the current semver based on your git history. Check the `installation guide <https://gitversion.net/docs/usage/cli/installation>`_.
   You can check the current semver value with::

     gitversion . /output json /showvariable SemVer

There is also a vscode plugin that integrates with the tool.

Requirements
~~~~~~~~~~~~

- **Unshallow**: The repository needs to be an unshallow clone. This means that the fetch-depth in GitHub Actions needs to be set to 0, for instance. Check with your build server to see how it can be configured appropriately. This is already configured in the checkout step inside the pipeline.

- **main branch**: The repository needs to have an existing local main or main branch. When you followed the *Getting Started* section in this document it will be there already.

.. warning::
   Protect your important branches from being pushed to directly!

GitHub Flow
-----------

The simplest approach. You have only one main branch defined in your repository.

1. Never commit directly on main, only via pull requests.
2. Create a feature branch from origin and commit changes
3. Do a PR to main and review it, make sure all CI checks pass. If not, fix and push to feature branch. If you have setup build-validation, a new push will rerun the CI check. Complete the PR and delete the feature branch. You can override the merge message with a nice message folowing the `gitchangelog` guidelines in case you want to be able to create a nice changelog.
4. You should now have a new version in the feed and a tag on main.
5. To start a new feature, run `git fetch --all` and

.. image:: ../_static/github-flow.png

Be sure to check this nice [infographic](https://guides.github.com/introduction/flow/) that explains the process. 

Gitversion will take care of the versioning for you and increment the patch version on every PR.
If you want to control the specific version for the next release you can do so in specifying the `next-version: <version>`
inside `GitVersion.yml`. Be sure to increment at least the minor part.

While this approach is the most basic usage of the gitversion tool you could add a bit more complexity.
For instance when you need artifacts to be tested in staging areas etc it might be better to work with alpha and/or beta
releases.

Github flow with release branches
---------------------------------

The github flow is able to support release branches. You can easily embed them in your workflow and consider them as long-lived
feature branches. The difference is that the ci/cd pipeline will release and publish beta releases of your packages. These can be used
to embed in downstream projects or deploys to test and staging environments.

1. Create a `releases/<version>` branch from main and consider it as you new main branch on which you will work during this release. To do this run
   
   .. code-block:: bash

     git checkout -b releases/<version>

   and set `<version>` as your new release version.

2. Bump the version to the new `semver` to keep the gitversion tool in control during PRs. You should use the `bumpversion` tool within this project. Please have a look at `bumpversion --help` for additional insights. 
TLDR, you need to run
   
   .. code-block:: bash

     bumpversion --commit --new-version <version> minor
  
   to set the version throughout the project.

3. Push the release branch and add policies on this branch as well, for instance build validation.

At this stage the release branch is the new main and can be used by yourself or developers to add features:

1. Create a short-lived feature branch from this release branch and start doing your work as before. Merge into the release branch via PR.
You can embed and test the resulting beta artifact in separate environments (test, staging,...)

2. Repeat until all features are done for this release. Then do a PR of the release branch to main. Gitversion takes the branch name `<version>` as the version to be published. 
Be sure to delete the release branch.

3. Start over with a new release branch.

Development branches
--------------------
The template also supports a `dev` branch on which you could work before moving towards a release branch if you want. Everything will work the same as on the release branch but it will publish alpha packages. The version logic can be controlled with the following section in `GitVersion.yml`:

.. code-block:: YAML

  branches:
    develop:
      increment: Patch

where the increment determines how the next version should be handled on alpha releases. You could also use `next-version=` with the latter having precedence over all other rules (so be careful).
If work is done, you can start a release branch from dev and tweak it for final release and do a PR from that one in main.

All things considered, the workflow you will apply in your project is up to you:

- No need for intermediate artifacts for testing/staging purposes? -> Use github flow
- Intermediate artifacts needed for testing/staging areas? -> Use github flow with `dev` branch and/or `releases` branch depending on how you organize your releases. 

There is no real need for a dev branch or alpha releases in most projects if you keep your features and releases cycles short. You don't need to put much thought in the new version (only at release branch creation and bumpversion command) but you already create versioned artifacts for testing purposes.

Gitflow
-------
In case you need the full complexity of a gitflow branching strategy you could do so as well.
It might be useful when you have multiple versions in production at the same time and you need to be able to hotfix/patch them separately.

In that case I suggest you read the gitversion tool documentation and tweak the gitversion settings to your liking.
