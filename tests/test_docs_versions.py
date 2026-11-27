"""Check that publishing docs preserves other releases and the stable root."""

import json
from pathlib import Path

import pytest

from scripts.assemble_docs import assemble_site


def test_publish_stable_and_candidate(tmp_path: Path) -> None:
    published = tmp_path / "published"
    (published / "3.0.0rc1").mkdir(parents=True)
    previous = b'<body><nav class="md-header__inner"></nav>RC1</body>'
    (published / "3.0.0rc1" / "index.html").write_bytes(previous)
    (published / "old-stable.html").write_text("Old stable page")
    (published / "CNAME").write_text("docs.example.org")

    build = tmp_path / "build"
    (build / "guide").mkdir(parents=True)
    stable_index = b'<body><div class="bd-header__inner"></div>2.1.1</body>'
    (build / "index.html").write_bytes(stable_index)
    (build / "guide" / "install.html").write_bytes(stable_index)
    stable_site = tmp_path / "stable"
    assemble_site(published, build, stable_site, "2.1.1", stable=True)

    assert not (stable_site / "old-stable.html").exists()
    assert (stable_site / "CNAME").read_text() == "docs.example.org"
    assert 'src="docs-versions.js"' in (stable_site / "index.html").read_text()
    assert (
        'src="../../docs-versions.js"'
        in (stable_site / "2.1.1" / "guide" / "install.html").read_text()
    )

    (build / "index.html").write_bytes(previous.replace(b"RC1", b"RC2"))
    site = tmp_path / "site"
    assemble_site(stable_site, build, site, "3.0.0rc2", stable=False)

    assert (site / "index.html").read_bytes() == (
        stable_site / "index.html"
    ).read_bytes()
    assert b"RC1" in (site / "3.0.0rc1" / "index.html").read_bytes()
    assert b"RC2" in (site / "3.0.0rc2" / "index.html").read_bytes()
    assert json.loads((site / "versions.json").read_text()) == {
        "stable": "2.1.1",
        "versions": ["3.0.0rc2", "3.0.0rc1", "2.1.1"],
    }
    with pytest.raises(ValueError, match="cannot become the stable docs"):
        assemble_site(
            site, build, tmp_path / "invalid", "3.0.0rc2", stable=True
        )


def test_publish_documentation_redirects(tmp_path: Path) -> None:
    published = tmp_path / "published"
    old_release = published / "3.0.0rc1" / "user"
    old_release.mkdir(parents=True)
    (old_release / "mcp.html").write_text("Historical MCP page")
    (old_release.parent / "index.html").write_text("Historical homepage")

    build = tmp_path / "build"
    (build / "mcp").mkdir(parents=True)
    (build / "user/examples").mkdir(parents=True)
    (build / "index.html").write_text("Current homepage")
    (build / "mcp/01-overview.html").write_text("MCP overview")
    (build / "user/examples/01-sn-curves.html").write_text("S–N curves")
    site = tmp_path / "site"
    assemble_site(published, build, site, "3.0.0", stable=True)

    for root in (site, site / "3.0.0"):
        mcp = (root / "user/mcp.html").read_text()
        assert 'href="../mcp/01-overview.html"' in mcp
        assert '"#connect-codex": ["../mcp/02-run-locally.html", ' in mcp
        assert '"#connect-a-stdio-client"' in mcp
        assert "location.search + anchor" in mcp
        example = (root / "user/examples/02-sn_curve.html").read_text()
        assert 'href="01-sn-curves.html"' in example
        assert not (root / "user/examples/mcp-hosted.html").exists()
    assert (
        site / "3.0.0rc1/user/mcp.html"
    ).read_text() == "Historical MCP page"
