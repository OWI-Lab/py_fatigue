"""Shared validation and preview rules for MCP results."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

Number = Annotated[FiniteFloat, Field(strict=True)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]


SERVER_INSTRUCTIONS = (
    "Use MPa for stress and mm for lengths. Use caller-supplied material inputs. "
    "For every plotting request, inspect plot_py_fatigue and use a native "
    "Py-Fatigue plot when available. Prefer the returned image or ECharts "
    "resource over rebuilding the chart from a preview. If the native tool "
    "does not cover the request, explain the gap before proposing custom plotting. "
    "Previews contain at most five rows. Offer CSV when available and export "
    "only on request. Reuse a synthetic specification to analyze all generated "
    "rows without copying samples from a preview. Preserve tensile load order "
    "for crack growth."
)


class InputModel(BaseModel):
    """Validate the JSON shape and reject unknown input fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


def preview_indices(row_count: int) -> list[int]:
    """Select at most five rows from the beginning and end of a result.

    Examples
    --------
    >>> preview_indices(8)
    [0, 1, 2, 6, 7]
    """
    if row_count <= 5:
        return list(range(row_count))
    return [0, 1, 2, row_count - 2, row_count - 1]
