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


class TestCylinder:
    """Test the functions related with the crack growth rules, i.e.:
    - Paris' law
    - Walker's law (TO BE IMPLEMENTED)
    """

    @given(
        outer_diameter=hy.floats(min_value=60.0, max_value=10000.0),
        initial_depth=hy.floats(min_value=1.0, max_value=10.0),
        height=hy.floats(min_value=10.0, max_value=1000),
    )
    @pytest.mark.parametrize("crack_position", [("internal"), ("external")])
    def test_hollow_cylinder(
        self,
        outer_diameter: float,
        initial_depth: float,
        height: float,
        crack_position: str,
    ):
        """Test the hollow cylinder geometry."""
        # Create the geometry
        thickness = initial_depth * 2.5 # max thickness can be 25, while
                                        # min outer_radius is 30
        geo = geometry.HollowCylinder(
            outer_diameter=outer_diameter,
            thickness=thickness,
            initial_depth=initial_depth,
            height=height,
            crack_position=crack_position,
        )

        assert geo.outer_diameter == outer_diameter
        assert geo.thickness == thickness
        assert geo.initial_depth == initial_depth
        assert geo.height == height
        assert geo.crack_position == crack_position
        assert (
            geo._id == "HOL_CYL_00"
            if geo.crack_position == "internal"
            else "HOL_CYL_01"
        )
        assert geo.geometry_factor.size == 0
        assert str(geo) == (
            F"HollowCylinder(\n  _id={geo._id}"
            + f",\n  initial_depth={initial_depth}"
            + f",\n  outer_diameter={outer_diameter}"
            + f",\n  thickness={thickness}"
            + f",\n  height={height}"
            + f",\n  width_to_depth_ratio={2.0}"
            + f",\n  crack_position={crack_position},\n)"
        )

        large_thick = outer_diameter * 2.0
        with pytest.raises(ValidationError):
            geometry.HollowCylinder(
                outer_diameter=outer_diameter,
                thickness=large_thick,
                initial_depth=initial_depth,
                height=height,
                crack_position=crack_position,
            )

        neg_thick = -outer_diameter / 2
        with pytest.raises(ValidationError):
            geometry.HollowCylinder(
                outer_diameter=outer_diameter,
                thickness=neg_thick,
                initial_depth=initial_depth,
                height=height,
                crack_position=crack_position,
            )

        with pytest.raises(ValidationError):
            geometry.HollowCylinder(
                outer_diameter=outer_diameter,
                thickness=thickness,
                initial_depth=initial_depth,
                height=height,
                crack_position="invalid",
            )

    @given(
        diameter=hy.floats(min_value=10.0, max_value=10000.0),
        initial_depth=hy.floats(min_value=1.0, max_value=9.0),
        height=hy.floats(min_value=10.0, max_value=1000),
    )
    def test_cylinder(
        self,
        diameter: float,
        initial_depth: float,
        height: float,
    ):
        """Test the solid cylinder geometry."""
        # Create the geometry
        geo = geometry.Cylinder(
            diameter=diameter,
            initial_depth=initial_depth,
            height=height,
        )

        assert geo.diameter == diameter
        assert geo.initial_depth == initial_depth
        assert geo.height == height
        assert geo._id == "FUL_CYL_00"
        assert geo.geometry_factor.size == 0
        assert str(geo) == (
            F"Cylinder(\n  _id={geo._id}"
            + f",\n  initial_depth={initial_depth}"
            + f",\n  diameter={diameter}"
            + f",\n  height={height},\n)"
        )

        neg_diam = -diameter
        with pytest.raises(ValidationError):
            geometry.Cylinder(
                diameter=neg_diam,
                initial_depth=initial_depth,
                height=height,
            )
        neg_height = -height
        with pytest.raises(ValidationError):
            geometry.Cylinder(
                diameter=diameter,
                initial_depth=neg_height,
                height=height,
            )
        
