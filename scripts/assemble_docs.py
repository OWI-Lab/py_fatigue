"""Assemble the GitHub Pages tree with preserved documentation versions."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?")
PICKER = "docs-versions.js"


def version_key(version: str) -> tuple[int, int, int, bool, int]:
    """Sort numeric versions with each final release above its candidates."""
    match = VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"Invalid docs version: {version}")
    major, minor, patch, candidate = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        candidate is None,
        int(candidate or 0),
    )


def add_version_picker(site: Path) -> None:
    """Add the shared picker to Zensical and Sphinx pages."""
    for page in site.rglob("*.html"):
        content = page.read_bytes()
        if b'id="docs-versions"' in content:
            continue
        if not any(
            marker in content
            for marker in (
                b'class="md-header__inner',
                b'class="bd-header__inner',
            )
        ):
            continue
        relative = os.path.relpath(site / PICKER, page.parent)
        script = f'<script id="docs-versions" src="{relative}" defer></script>'
        page.write_bytes(
            content.replace(b"</body>", script.encode() + b"\n</body>", 1)
        )


def assemble_site(
    published: Path, build: Path, site: Path, version: str, *, stable: bool
) -> None:
    """Replace one version, optionally mirror it at the root, and index both."""
    key = version_key(version)
    if stable and not key[3]:
        raise ValueError("A release candidate cannot become the stable docs.")
    if not (build / "index.html").is_file():
        raise ValueError(f"Missing build homepage in {build}")

    entries = list(published.iterdir())
    catalog = published / "versions.json"
    stable_version = (
        json.loads(catalog.read_text())["stable"] if catalog.is_file() else None
    )
    site.mkdir()

    for entry in entries:
        if entry.name in {".git", version, "versions.json"}:
            continue
        if stable and entry.name not in {".nojekyll", "CNAME"}:
            if VERSION.fullmatch(entry.name) is None:
                continue
        destination = site / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
        else:
            shutil.copy2(entry, destination)

    shutil.copytree(build, site / version)
    if stable:
        shutil.copytree(build, site, dirs_exist_ok=True)
        stable_version = version

    versions = sorted(
        (
            entry.name
            for entry in site.iterdir()
            if VERSION.fullmatch(entry.name)
            and (entry / "index.html").is_file()
        ),
        key=version_key,
        reverse=True,
    )
    (site / "versions.json").write_text(
        json.dumps({"stable": stable_version, "versions": versions}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    picker = (
        Path(__file__).resolve().parents[1] / "docs" / "_static" / "js" / PICKER
    )
    shutil.copy2(picker, site / PICKER)
    add_version_picker(site)


def main() -> None:
    """Read deployment paths and the version from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("published", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("site", type=Path)
    parser.add_argument("version")
    parser.add_argument("--stable", action="store_true")
    args = parser.parse_args()
    assemble_site(
        args.published, args.build, args.site, args.version, stable=args.stable
    )


if __name__ == "__main__":
    main()
