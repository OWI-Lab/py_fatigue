import json
from invoke import task

@task(help={"organization": "Azure DevOps organization.",
            "project": "Azure DevOps project."})
def config(c, organization=None, project=None):
    """Configure az devops cli, uses values from invoke.yaml by default."""
    if organization is None:
        organization = c.ado.organization
    if project is None:
        project = c.ado.project
    cmd = f"az devops configure --defaults organization=\"{organization}\" project=\"{project}\""
    print(cmd)
    c.run(cmd)

@task(help={"name": "Name of repository.",
            "open": "Open the repository page in your web browser."})
def repo_create(c, name=None, open=False):
    """Create az devops repository, uses values from invoke.yaml by default."""
    if name is None:
        name = c.ado.repository.name
    cmd = f"az repos create --name=\"{name}\""
    if open:
        cmd += " --open"
    c.run(cmd, echo=True)


# TODO: Automate git setup

# @task(help={"name": "Name of repository."})
# def add_origin(c, name=None):
#     """Show az devops repository, uses values from invoke.yaml by default."""
#     if name is None:
#         name = c.ado.repository.name
#     # Get info from repo
#     cmd = f"az repos show --repository=\"{name}\" --query sshUrl"
#     url = c.run(cmd, echo=True)
#     print(url)

#     c.run("git init", echo=True)
#     c.run(f"git remote add origin {url}", echo=True)
