import re
import sys

# Define the regex pattern for version strings
_REGEX = "".join(
    [
        r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)",
        r"(?:\-(?P<release>.*)\.(?P<num>\d+))?",
    ]
)


def parse_version(version_str):
    """Parse the version string using the regex pattern."""
    match = re.match(_REGEX, version_str)
    if not match:
        raise ValueError(f"Invalid version string: {version_str}")
    return match.groupdict()


def update_version_file(version_str=None, bump_type=None):
    """Update the ./VERSION file with the new version or bump the current version."""
    if bump_type:
        with open("./py_fatigue/version.py", "r") as f:
            for line in f:
                if line.startswith("__version__: str ="):
                    current_version = line.split('"')[1]
                    break
            else:
                raise ValueError("Current version not found in version.py")

        version_info = parse_version(current_version)
        if bump_type == "major":
            version_info["major"] = str(int(version_info["major"]) + 1)
            version_info["minor"] = "0"
            version_info["patch"] = "0"
        elif bump_type == "minor":
            version_info["minor"] = str(int(version_info["minor"]) + 1)
            version_info["patch"] = "0"
        elif bump_type == "patch":
            version_info["patch"] = str(int(version_info["patch"]) + 1)
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

        version_str = "{major}.{minor}.{patch}".format(**version_info)

    if not version_str:
        raise ValueError("Either version_str or bump_type must be provided")

    with open("./VERSION", "w") as f:
        f.write(version_str)
        # add also a new line
        f.write("\n")


def update_version_in_file(version_str, version_file, version_var):
    """Update the version variable in the specified file."""
    with open(version_file, "r") as f:
        lines = f.readlines()

    with open(version_file, "w") as f:
        for line in lines:
            if line.startswith(version_var):
                f.write(f'{version_var} "{version_str}"\n')
            else:
                f.write(line)


def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print(
            "Usage: python bumpversion.py <version> or python bumpversion.py <bump_type>"
        )
        sys.exit(1)

    version_str = None
    bump_type = None

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg in ["major", "minor", "patch"]:
            bump_type = arg
        else:
            version_str = arg
            parse_version(version_str)  # Validate the version string
    elif len(sys.argv) == 3:
        print(
            "Invalid usage. Provide either a version string or a bump type, not both."
        )
        sys.exit(1)

    update_version_file(version_str, bump_type)
    with open("./VERSION", "r") as f:
        new_version_str = f.read().strip()

    update_version_in_file(
        new_version_str, "./py_fatigue/version.py", "__version__: str ="
    )
    update_version_in_file(
        new_version_str, "./tests/conftest.py", "VERSION: str ="
    )
    update_version_in_file(
        new_version_str, "./pyproject.toml", "version ="
    )
    print(f"Version updated to {new_version_str}")


if __name__ == "__main__":
    main()
