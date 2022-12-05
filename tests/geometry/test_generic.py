# -*- coding: utf-8 -*-

r"""The following tests are meant to assess the correct behavior of the
damage calculation methods in the stress-life approach.
"""


# Packages from the Python Sstandard Library
import os
import pytest
import sys

# Packages from non-standard libraries
from pydantic import ValidationError
from hypothesis import given, strategies as hy

# Local imports
import py_fatigue.geometry as geometry

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)
# Non-standard imports


class TestGeneric:
    """Test the functions related with the implementation of the generic
    geometry class.
    """

    @given(
        initial_depth=hy.floats(min_value=1.0, max_value=10.0),
    )
    def test_infinite_surface(
        self,
        initial_depth: float,
    ):
        """Test the hollow cylinder geometry."""
        # Create the geometry
        geo = geometry.InfiniteSurface(initial_depth=initial_depth)   
        assert geo.initial_depth == initial_depth
        assert geo._id == "INF_SUR_00"
        assert str(geo) == (
            F"InfiniteSurface(\n  _id={geo._id}"
            + f",\n  initial_depth={initial_depth},\n)"
        )

        neg_depth = -initial_depth
        with pytest.raises(ValidationError):
            geometry.InfiniteSurface(initial_depth=neg_depth)
