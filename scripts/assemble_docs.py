"""Assemble the GitHub Pages tree with preserved documentation versions."""

import argparse
from html import escape
import json
import os
import re
import shutil
from pathlib import Path

VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?")
PICKER = "docs-versions.js"
EXAMPLE_MOVES = {
    "02-sn_curve": "01-sn-curves",
    "03-cg_curve": "02-crack-growth-curves",
    "04-cycle_counting": "03-count-stress-cycles",
    "05-cycle_count_sum": "04-combine-cycle-counts",
    "06-mean_stress": "05-correct-for-mean-stress",
    "07-damage": "06-stress-life-damage",
    "08-ordered-crack-growth": "07-ordered-crack-growth",
    "01-notebook_setup": "08-example-notebooks",
}
MCP_MOVES = {
    "user/mcp.html": "mcp/01-overview.html",
    "user/examples/mcp-hosted.html": "mcp/04-analyze-stress-history.html",
    "user/examples/mcp-fatigue.html": "mcp/05-analyze-local-csv.html",
}
MCP_MOVED_FRAGMENTS = {
    "user/mcp.html": {
        "connect-codex": ("02-run-locally", "connect-a-stdio-client"),
        **{
            anchor: ("06-tool-reference", anchor)
            for anchor in (
                "count-cycles",
                "calculate-miner-damage",
                "simulate-ordered-crack-growth",
                "use-local-csv-files",
            )
        },
    },
    "user/examples/mcp-hosted.html": {
        anchor: ("03-connect-hosted", anchor)
        for anchor in (
            "load-your-token",
            "connect-your-client",
            "codex-cli",
            "claude-code",
        )
    },
}


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


def add_doc_redirects(root: Path) -> None:
    """Preserve moved paths and deep links within the same docs version."""
    moves = {
        **{
            f"user/examples/{old}.html": f"user/examples/{new}.html"
            for old, new in EXAMPLE_MOVES.items()
        },
        **MCP_MOVES,
    }
    for old, new in moves.items():
        alias = root / old
        target = root / new
        if alias.exists() or not target.is_file():
            continue
        relative = os.path.relpath(target, alias.parent)
        fragments = {
            f"#{old_anchor}": [
                os.path.relpath(root / "mcp" / f"{page}.html", alias.parent),
                f"#{anchor}",
            ]
            for old_anchor, (page, anchor) in MCP_MOVED_FRAGMENTS.get(
                old, {}
            ).items()
        }
        href = escape(relative, quote=True)
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(
            '<!doctype html>\n<html lang="en"><meta charset="utf-8">\n'
            '<meta name="robots" content="noindex"><title>Page moved</title>\n'
            f'<p>This page moved. <a href="{href}">Open the new page</a>.</p>\n'
            f"<script>const links = {json.dumps(fragments)};\n"
            "const [path, anchor] = links[location.hash] ?? "
            f"[{json.dumps(relative)}, location.hash];\n"
            "location.replace(path + location.search + anchor);</script>\n"
            "</html>\n",
            encoding="utf-8",
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
    add_doc_redirects(site / version)
    if stable:
        shutil.copytree(build, site, dirs_exist_ok=True)
        add_doc_redirects(site)
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
