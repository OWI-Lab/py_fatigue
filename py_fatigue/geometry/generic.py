# Packages from the Python Standard Library
import abc
from functools import wraps

# Packages from non-standard libraries
from pydantic.dataclasses import dataclass
from pydantic.types import PositiveFloat

# Local imports


def kwargs_only(cls):
    """Decorator to make a class's __init__ method accept only keyword
    arguments.

    This is useful for dataclasses, which by default accept positional
    arguments.
    """
    old_init = cls.__init__

    @wraps(old_init)
    def new_init(self, *args, **kwargs):
        if args:
            raise TypeError("This class only accepts keyword arguments.")
        old_init(self, **kwargs)

    cls.__init__ = new_init
    return cls


# @kwargs_only
@dataclass(repr=False)
class CrackGeometryMixin:
    """Mixin class to define the abstract methods for the
    AbstractCrackGeometry class.

    Parameters
    ----------
    initial_depth : float
        The initial depth of the crack.
    """

    initial_depth: PositiveFloat
    # crack_depth_: Optional[TypedArray] = None
    # count_cycle_: Optional[TypedArray] = None
    # sif_: Optional[TypedArray] = None
    # final_cycles_: Optional[PositiveFloat] = None


class AbstractCrackGeometry(CrackGeometryMixin, metaclass=abc.ABCMeta):
    """Abstract class to initalise and store the crack geometry data."""

    def __str__(self) -> str:

        str_ = "\n".join(
            [
                f"{self.__class__.__name__}(",
                f"  _id={self._id},",
                f"  initial_depth={self.initial_depth},",
                ")",
            ]
        )
        return str_

    def __repr__(self) -> str:
        return self.__str__()

    @property
    @abc.abstractmethod
    def _id(self) -> property:
        """Get the ID of the crack geometry."""

    # @property
    # def crack_depth(self) -> Optional[TypedArray]:
    #     """Get the crack_depth. This is the result of an analysis."""
    #     return self.crack_depth_

    # @crack_depth.setter
    # def crack_depth(self, value: TypedArray) -> None:
    #     """Set the crack_depth. This is the result of an analysis."""
    #     self.crack_depth_ = value

    # @crack_depth.deleter
    # def crack_depth(self) -> None:
    #     """Delete the crack_depth."""
    #     del self.crack_depth_

    # @property
    # def count_cycle(self) -> Optional[TypedArray]:
    #     """Get the count_cycle. This is the result of an analysis."""
    #     return self.count_cycle_

    # @count_cycle.setter
    # def count_cycle(self, value: TypedArray) -> None:
    #     """Set the count_cycle. This is the result of an analysis."""
    #     self.count_cycle_ = value

    # @count_cycle.deleter
    # def count_cycle(self) -> None:
    #     """Delete the count_cycle."""
    #     del self.count_cycle_

    # @property
    # def sif(self) -> Optional[TypedArray]:
    #     """Get the sif. This is the result of an analysis."""
    #     return self.sif_

    # @sif.setter
    # def sif(self, value: TypedArray) -> None:
    #     """Set the sif. This is the result of an analysis."""
    #     self.sif_ = value

    # @sif.deleter
    # def sif(self) -> None:
    #     """Delete the sif."""
    #     del self.sif_

    # @property
    # def final_cycles(self) -> Optional[PositiveFloat]:
    #     """Get the final_cycles. This is the result of an analysis."""
    #     return self.final_cycles_

    # @final_cycles.setter
    # def final_cycles(self, value: PositiveFloat) -> None:
    #     """Set the final_cycles. This is the result of an analysis."""
    #     self.final_cycles_ = value

    # @final_cycles.deleter
    # def final_cycles(self) -> None:
    #     """Delete the final_cycles."""
    #     del self.final_cycles_


@dataclass(repr=False)
class InfiniteSurface(AbstractCrackGeometry):
    """Class to initalise and store the crack geometry data on a
    flat infinite surface, as per standard Paris' law.

    Parameters
    ----------
    initial_depth : float
        The initial depth of the crack.
    """

    _id = property(lambda _: "INF_SUR_00")

    # @property
    # def geometry_factor(self) -> np.ndarray:
    #     """Get the geometric factor. This should be a function of the crack
    #     size. The default value is an empty array."""
    #     return np.empty(0)

    # @property
    # def crack_depth(self) -> np.ndarray:
    #     """Get the crack_depth. This is the result of an analysis."""
    #     return np.empty(0)

    # @property
    # def stress_intensity_factor(self) -> np.ndarray:
    #     """Get the stress intensity factor. This is the result of an
    #     analysis."""
    #     return np.empty(0)

    # @property
    # def count_cycle(self) -> np.ndarray:
    #     """Get the growth cycles. This is the result of an analysis."""
    #     return np.empty(0)

    # @property
    # def final_cycles(self) -> Optional[PositiveFloat]:
    #     """Get the final cycles. This is the result of an analysis."""

    # @property
    # def final_depth(self) -> Optional[PositiveFloat]:
    #     """Get the final crack depth. This is the result of an analysis."""
