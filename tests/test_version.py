import pytest
from hypothesis import given, strategies as hy

from py_fatigue.version import __version__, parse_version

from .conftest import VERSION


def test_version():
    assert __version__ == VERSION


@pytest.mark.success
@given(
    major=hy.integers(min_value=0),
    minor=hy.integers(min_value=0),
    patch=hy.integers(min_value=0),
)
def test_parse_version(major, minor, patch):
    version = f"{major}.{minor}.{patch}"
    actual = parse_version(version)

    assert actual.major == major
    assert actual.minor == minor
    assert actual.patch == patch
    assert not actual.release
    assert not actual.num


@pytest.mark.success
@given(
    major=hy.integers(min_value=0),
    minor=hy.integers(min_value=0),
    patch=hy.integers(min_value=0),
    release=hy.text(
        min_size=1, alphabet=hy.characters(whitelist_categories=["L", "N"])
    ),
    num=hy.integers(min_value=0),
)
def test_parse_version_with_release_and_num(major, minor, patch, release, num):
    version = f"{major}.{minor}.{patch}-{release}.{num}"
    actual = parse_version(version)

    assert actual.major == major
    assert actual.minor == minor
    assert actual.patch == patch
    assert actual.release == release
    assert actual.num == num


@pytest.mark.error
@given(
    major=hy.integers(max_value=-1),
    minor=hy.integers(max_value=-1),
    patch=hy.integers(max_value=-1),
)
def test_negative_version(major, minor, patch):
    with pytest.raises(ValueError):
        version = f"{major}.{minor}.{patch}"
        parse_version(version)
