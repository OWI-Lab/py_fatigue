# Artifact feeds

Artifacts Feeds are organizational constructs that allow you to store, manage, and group your packages and control who to share it with. 
Feeds are not package-type dependent. You can store all the following package types in a single feed: 
npm, NuGet, Maven, Python, and Universal packages.

## Create feed
If there is no feed created yet or you just want a separate one you can follow this [guide](https://docs.microsoft.com/en-us/azure/devops/artifacts/quickstarts/python-packages?view=azure-devops) to set one up. 

In general you don't need upstream sources enabled as `poetry` will default to pypi. Mirroring on ADO tends to be slow as well and if accidentally a wrong package gets imported the whole feed is immutable and can't be fixed.

Keep in mind that in the python ecosystem _packages are expected to be unique up to name and version_, so two wheels with the same package name and version are treated as indistinguishable by pip. This is a deliberate feature of the package metadata, and not likely to change. So while poetry in itself can support this by specifying the `source` directly

    poetry add --source <SOURCE> <PACKAGE>

other tools, like pip, do not support this and probably never will. A simple fix to mitigate this potential issue:

- Use a **prefix** for your package name so your package is unique in the python ecosystem. Be sure to check [pypi](https://pypi.org/) if your package name already exists. For instance, `kpv-superb-package`. It also has the benefit of clearly indicating that a package is private. 

You could rely on poetry combined with the `source` specification to do the right thing. This means that `poetry install` is the only valid way to fetch the correct packages. Even `poetry export` will not adhere to the source as it generates a pip compliant `requirements.txt` which does not support index prioritization. 

## Configure poetry

When there is a need to incorporate packages from a (private) ADO feed (i.e. an extra index), poetry needs to know where and how to connect to it.
The `pyproject.toml` file needs a `[[tool.poetry.source]]` section to specify an additional index available for packages. Poetry will still use pypi as default and first choice. 

    [[tool.poetry.source]]
    name = "<SOURCE>"    
    url = "https://pkgs.dev.azure.com/<ORG>/<PRJ>/_packaging/<FEED_NAME>/pypi/simple/"
    secondary = true

- `<SOURCE>`: A name for the source, you can use `<FEED_NAME>` for transparency.
- `<FEED_NAME>`: The name of the artifact feed you want to use. 
- `<ORG>`: Your Azure DevOps organization.
- `<PRJ>`: Your Azure Devops project.

:::{Note}
To obtain the url you can check in ADO on the Artifacts page -> Connect to feed -> choose pip -> copy the url.
:::

In most case the Azure Artifact feed will be private so setting up credentials is required. See the next section for information on how to set this up.

## Set credentials for Azure DevOps artifact feeds

To be able to authenticate you need to generate a personal token.
In ADO you can generate a Personal Access Token linked with your user account, see [this guide](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=preview-page) for the setup process. 

It's best to setup a _least privilege tokens with an expiration date_. To read feeds you need Packaging -> Read scope as a minimum. Take note of the PAT as you will need it to setup the authentication mechanism for the tools in this project. Don;t forget that you will need to repeat this auth setup when your PAT expires in the future.

### poetry

Run the following poetry command (preferred) where `username` is your Azure DevOps account email and `password` is the PAT token. The `<SOURCE>` is the name of the source section in `pyproject.toml`. If you setup an keyring on your system it will use that or fallback to a text based config located at `~/.config/pypoetry/auth.toml`.

    > poetry config http-basic.<SOURCE> username password

or manual configure `~/.config/pypoetry/auth.toml` as:

    [http-basic]
    [http-basic.<SOURCE>]
    username = <Azure Devops account email>
    password = <Personal Access Token>

See `[[tool.poetry.source]]` section in `pyproject.toml` to find the value for `<SOURCE>`. It is the name that was given to the Azure Artifacts feed URL (`name = "<SOURCE>"`). 

As an alternative you can use environment variables as well.

    export POETRY_HTTP_BASIC_<SOURCE>_USERNAME=<username>
    export POETRY_HTTP_BASIC_<SOURCE>_PASSWORD=<PAT>

Now `poetry install --no-root` and other poetry commands should work with the private feed. Keep in mind that you will need to recreate your PAT when it expires.

:::{Note}
In case you have problems with authentication check
1. Remove `.netrc` from your home dir
2. Check if poetry config does not have repositories specified (`poetry config --list`)
:::

### twine
 Used to upload packages (optional), normally covered by pipeline. You will need Read & Write packaging scope for your PAT. 
    
Configure `~/.pypirc` as:

    [distutils]
    index-servers = <SOURCE>

    [<SOURCE>]
    repository = https://pkgs.dev.azure.com/<ORG>/<PRJ>/_packaging/<FEED_NAME>/pypi/upload/
    username = <username>
    password = <Personal Access Token>

or configure the environment variables:    

    export TWINE_USERNAME=<username> - the username to use for authentication to the repository.
    export TWINE_PASSWORD=<Personal Access Token> - the PAT to use for authentication to the repository. 

Now run `twine upload -r <SOURCE> dist/*` to upload all packages in your dist folder. 

## pip 
Python package install, normally not needed as poetry takes care of that (optional)

Analog approach with `.netrc` file. See [here](https://pip.pypa.io/en/stable/user_guide/#netrc-support) for more information.

Keep in mind that index order is not guaranteed!
