# -*- coding: utf-8 -*-

r"""Comprehensive tests for the geometry.generic module.

This module tests all classes and functions in the generic geometry module:
- kwargs_only decorator
- CrackGeometryMixin dataclass
- AbstractCrackGeometry abstract class
- InfiniteSurface concrete implementation
"""

# Packages from the Python Standard Library
import os
import sys
import pytest

# Packages from non-standard libraries
from pydantic import ValidationError
from hypothesis import given, strategies as hy

# Local imports
import py_fatigue.geometry as geometry
from py_fatigue.geometry.generic import (
    kwargs_only,
    CrackGeometryMixin,
    AbstractCrackGeometry,
    InfiniteSurface
)

PROJECT_PATH = os.path.dirname(os.getcwd())
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)


class TestKwargsOnlyDecorator:
    """Test the kwargs_only decorator functionality."""

    def test_kwargs_only_decorator_basic(self):
        """Test that kwargs_only decorator forces keyword-only arguments."""

        @kwargs_only
        class TestClass:
            def __init__(self, a, b=2):
                self.a = a
                self.b = b

        # Should work with keyword arguments
        obj = TestClass(a=1, b=3)
        assert obj.a == 1
        assert obj.b == 3

        # Should raise TypeError with positional arguments
        with pytest.raises(TypeError, match="only accepts keyword arguments"):
            TestClass(1, 2)

    def test_kwargs_only_decorator_with_dataclass(self):
        """Test kwargs_only decorator with dataclass."""
        from dataclasses import dataclass

        @kwargs_only
        @dataclass
        class TestDataClass:
            x: float
            y: float = 1.0

        # Should work with keyword arguments
        obj = TestDataClass(x=5.0, y=2.0)
        assert obj.x == 5.0
        assert obj.y == 2.0

        # Should raise TypeError with positional arguments
        with pytest.raises(TypeError, match="only accepts keyword arguments"):
            TestDataClass(5.0, 2.0)

    def test_kwargs_only_preserves_original_functionality(self):
        """Test that kwargs_only preserves the original __init__ functionality."""

        @kwargs_only
        class TestClass:
            def __init__(self, required, optional=None):
                self.required = required
                self.optional = optional

        obj = TestClass(required="test", optional="value")
        assert obj.required == "test"
        assert obj.optional == "value"

        # Should work with only required parameter
        obj2 = TestClass(required="test2")
        assert obj2.required == "test2"
        assert obj2.optional is None


class TestCrackGeometryMixin:
    """Test the CrackGeometryMixin dataclass."""

    def test_crack_geometry_mixin_initialization(self):
        """Test basic initialization of CrackGeometryMixin."""
        mixin = CrackGeometryMixin(initial_depth=5.0)
        assert mixin.initial_depth == 5.0

    @given(initial_depth=hy.floats(min_value=0.1, max_value=100.0))
    def test_crack_geometry_mixin_positive_depth(self, initial_depth):
        """Test that CrackGeometryMixin accepts positive depths."""
        mixin = CrackGeometryMixin(initial_depth=initial_depth)
        assert mixin.initial_depth == initial_depth

    def test_crack_geometry_mixin_negative_depth_validation(self):
        """Test that negative depths raise ValidationError."""
        with pytest.raises(ValidationError):
            CrackGeometryMixin(initial_depth=-1.0)

    def test_crack_geometry_mixin_zero_depth_validation(self):
        """Test that zero depth raises ValidationError."""
        with pytest.raises(ValidationError):
            CrackGeometryMixin(initial_depth=0.0)


class TestAbstractCrackGeometry:
    """Test the AbstractCrackGeometry abstract class."""

    def test_abstract_crack_geometry_cannot_instantiate(self):
        """Test that AbstractCrackGeometry cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractCrackGeometry(initial_depth=5.0)

    def test_abstract_crack_geometry_concrete_implementation(self):
        """Test that concrete implementations work correctly."""

        class ConcreteGeometry(AbstractCrackGeometry):
            @property
            def _id(self):
                return "TEST_GEO_01"

        geo = ConcreteGeometry(initial_depth=3.0)
        assert geo.initial_depth == 3.0
        assert geo._id == "TEST_GEO_01"

    def test_abstract_crack_geometry_str_representation(self):
        """Test string representation of AbstractCrackGeometry."""

        class ConcreteGeometry(AbstractCrackGeometry):
            @property
            def _id(self):
                return "TEST_GEO_02"

        geo = ConcreteGeometry(initial_depth=7.5)
        str_repr = str(geo)

        assert "ConcreteGeometry(" in str_repr
        assert "_id=TEST_GEO_02" in str_repr
        assert "initial_depth=7.5" in str_repr

    def test_abstract_crack_geometry_repr_equals_str(self):
        """Test that __repr__ equals __str__."""

        class ConcreteGeometry(AbstractCrackGeometry):
            @property
            def _id(self):
                return "TEST_GEO_03"

        geo = ConcreteGeometry(initial_depth=2.3)
        assert repr(geo) == str(geo)

    def test_abstract_crack_geometry_must_implement_id(self):
        """Test that concrete classes must implement _id property."""

        class IncompleteGeometry(AbstractCrackGeometry):
            pass  # Missing _id implementation

        with pytest.raises(TypeError):
            IncompleteGeometry(initial_depth=1.0)

    def test_abstract_crack_geometry_id_property_abstract(self):
        """Test that _id is correctly marked as abstract."""

        # Check that _id is in the abstract methods
        abstract_methods = AbstractCrackGeometry.__abstractmethods__
        assert '_id' in abstract_methods


class TestInfiniteSurface:
    """Test the InfiniteSurface concrete implementation."""

    @given(initial_depth=hy.floats(min_value=0.1, max_value=100.0))
    def test_infinite_surface_basic_properties(self, initial_depth):
        """Test basic properties of InfiniteSurface."""
        geo = InfiniteSurface(initial_depth=initial_depth)
        assert geo.initial_depth == initial_depth
        assert geo._id == "INF_SUR_00"

    def test_infinite_surface_id_property(self):
        """Test that InfiniteSurface has correct ID."""
        geo = InfiniteSurface(initial_depth=1.0)
        assert geo._id == "INF_SUR_00"

        # Test that _id is a property, not just an attribute
        assert isinstance(type(geo)._id, property)

    def test_infinite_surface_str_representation(self):
        """Test string representation of InfiniteSurface."""
        initial_depth = 5.0
        geo = InfiniteSurface(initial_depth=initial_depth)

        expected_str = (
            f"InfiniteSurface(\n"
            f"  _id={geo._id},\n"
            f"  initial_depth={initial_depth},\n"
            f")"
        )

        assert str(geo) == expected_str

    def test_infinite_surface_repr_representation(self):
        """Test repr representation of InfiniteSurface."""
        geo = InfiniteSurface(initial_depth=3.7)
        assert repr(geo) == str(geo)

    def test_infinite_surface_validation_negative_depth(self):
        """Test that InfiniteSurface validates negative depth."""
        with pytest.raises(ValidationError):
            InfiniteSurface(initial_depth=-5.0)

    def test_infinite_surface_validation_zero_depth(self):
        """Test that InfiniteSurface validates zero depth."""
        with pytest.raises(ValidationError):
            InfiniteSurface(initial_depth=0.0)

    def test_infinite_surface_inheritance(self):
        """Test that InfiniteSurface correctly inherits from AbstractCrackGeometry."""
        geo = InfiniteSurface(initial_depth=2.0)

        # Should be instance of both classes
        assert isinstance(geo, InfiniteSurface)
        assert isinstance(geo, AbstractCrackGeometry)
        assert isinstance(geo, CrackGeometryMixin)

    @given(
        initial_depth_1=hy.floats(min_value=0.1, max_value=50.0),
        initial_depth_2=hy.floats(min_value=0.1, max_value=50.0)
    )
    def test_infinite_surface_equality_based_on_depth(self, initial_depth_1, initial_depth_2):
        """Test that two InfiniteSurface objects with same depth have same properties."""
        geo1 = InfiniteSurface(initial_depth=initial_depth_1)
        geo2 = InfiniteSurface(initial_depth=initial_depth_2)

        # Both should have the same _id
        assert geo1._id == geo2._id == "INF_SUR_00"

        # Their depths should match what was input
        assert geo1.initial_depth == initial_depth_1
        assert geo2.initial_depth == initial_depth_2

    def test_infinite_surface_edge_cases(self):
        """Test edge cases for InfiniteSurface."""
        # Very small positive depth
        geo_small = InfiniteSurface(initial_depth=1e-10)
        assert geo_small.initial_depth == 1e-10
        assert geo_small._id == "INF_SUR_00"

        # Very large depth
        geo_large = InfiniteSurface(initial_depth=1e6)
        assert geo_large.initial_depth == 1e6
        assert geo_large._id == "INF_SUR_00"

    def test_infinite_surface_dataclass_behavior(self):
        """Test that InfiniteSurface behaves as expected dataclass."""
        geo = InfiniteSurface(initial_depth=4.2)

        # Should have dataclass fields
        assert hasattr(geo, '__dataclass_fields__')
        assert 'initial_depth' in geo.__dataclass_fields__

        # Field should exist and have annotation
        field = geo.__dataclass_fields__['initial_depth']
        assert field.type is not None


class TestIntegration:
    """Integration tests for the geometry.generic module."""

    def test_geometry_module_exports(self):
        """Test that geometry module exports expected classes."""
        assert hasattr(geometry, 'InfiniteSurface')
        assert hasattr(geometry, 'AbstractCrackGeometry')

    def test_abc_metaclass_functionality(self):
        """Test that ABC metaclass works correctly."""
        # InfiniteSurface should be instantiable as it implements all abstract methods
        geo = InfiniteSurface(initial_depth=1.0)
        assert isinstance(geo, AbstractCrackGeometry)

        # An incomplete implementation should fail
        class IncompleteGeometry(AbstractCrackGeometry):
            pass

        with pytest.raises(TypeError):
            IncompleteGeometry(initial_depth=1.0)

    def test_pydantic_integration(self):
        """Test that pydantic integration works correctly."""
        # Valid case
        geo = InfiniteSurface(initial_depth=5.0)
        assert isinstance(geo.initial_depth, float)

        # Invalid cases should raise ValidationError
        with pytest.raises(ValidationError):
            InfiniteSurface(initial_depth="not_a_number")

        with pytest.raises(ValidationError):
            InfiniteSurface(initial_depth=-1.0)

    @given(depths=hy.lists(hy.floats(min_value=0.1, max_value=100.0), min_size=1, max_size=10))
    def test_multiple_geometry_objects(self, depths):
        """Test creating multiple geometry objects."""
        geometries = [InfiniteSurface(initial_depth=depth) for depth in depths]

        # All should be valid
        assert len(geometries) == len(depths)

        # All should have correct properties
        for geo, depth in zip(geometries, depths):
            assert geo.initial_depth == depth
            assert geo._id == "INF_SUR_00"
            assert isinstance(geo, AbstractCrackGeometry)

    def test_commented_properties_not_implemented(self):
        """Test that commented-out properties are not implemented."""
        geo = InfiniteSurface(initial_depth=1.0)

        # These properties are commented out in the source
        # so they should not exist
        assert not hasattr(geo, 'crack_depth')
        assert not hasattr(geo, 'geometry_factor')
        assert not hasattr(geo, 'stress_intensity_factor')
        assert not hasattr(geo, 'count_cycle')
        assert not hasattr(geo, 'final_cycles')
        assert not hasattr(geo, 'final_depth')
