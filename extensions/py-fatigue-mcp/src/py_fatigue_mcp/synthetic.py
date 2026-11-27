"""Bounded, reproducible inputs from py_fatigue.testing."""

import math
from threading import Lock
from typing import Annotated, Literal

from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, StrictBool

from py_fatigue.testing import get_random_data, get_sampled_time

from .csv_io import CsvExport, write_csv_rows
from .models import InputModel, Nonnegative, Number, Positive, preview_indices

MAX_SYNTHETIC_SAMPLES = 100_000
MAX_LOAD_BLOCKS = 10_000
LOAD_COLUMNS = ("minimum", "maximum", "cycles")
SAMPLE_COLUMNS = ("time", "stress")
RANDOM_LOCK = Lock()

DistributionName = Literal[
    "normal", "uniform", "weibull", "rayleigh", "lognormal", "exponential"
]
DISTRIBUTION_PARAMETERS = {
    "normal": {"loc", "scale"},
    "uniform": {"low", "high"},
    "weibull": {"a"},
    "rayleigh": {"scale"},
    "lognormal": {"mean", "sigma"},
    "exponential": {"scale"},
}


class RandomSettings(InputModel):
    """Seed and documented parameters for a supported distribution."""

    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 2
    random_type: DistributionName = "normal"
    parameters: Annotated[dict[str, Number], Field(max_length=2)] = Field(
        default_factory=dict
    )


class SyntheticHistory(RandomSettings):
    """Sample a stress record in seconds and MPa."""

    kind: Literal["synthetic_history"]
    fs: Positive
    duration: Positive
    start: Nonnegative = 0.0
    minimum: Number
    maximum: Number


class SyntheticLoads(RandomSettings):
    """Generate ordered tensile blocks with a varying maximum in MPa."""

    kind: Literal["synthetic_loads"]
    block_count: Annotated[int, Field(ge=2, le=MAX_LOAD_BLOCKS)]
    minimum: Nonnegative
    lowest_peak: Positive
    highest_peak: Positive
    cycles_per_block: Positive


SyntheticInput = Annotated[
    SyntheticHistory | SyntheticLoads, Field(discriminator="kind")
]


class SampleRow(BaseModel):
    """One sample time in seconds and stress in MPa."""

    time: Number
    stress: Number


class TensileLoad(InputModel):
    """Ordered tensile stress extrema in MPa and a fractional cycle count."""

    minimum: Nonnegative
    maximum: Positive
    cycles: Positive


class SyntheticSummary(BaseModel):
    """Reusable specification and a bounded preview of generated data."""

    specification: SyntheticInput
    row_count: int
    preview: list[SampleRow | TensileLoad]
    export: CsvExport | None = None


def scaled_random(
    time: NDArray[np.float64],
    settings: RandomSettings,
    minimum: float,
    maximum: float,
) -> NDArray[np.float64]:
    """Use Py-Fatigue's generator without changing the caller's RNG state."""
    span = maximum - minimum
    if not math.isfinite(span) or span <= 0:
        raise ToolError(
            "Synthetic maximum must exceed minimum by a finite span."
        )
    unexpected = (
        settings.parameters.keys()
        - DISTRIBUTION_PARAMETERS[settings.random_type]
    )
    if unexpected:
        raise ToolError(
            f"Unsupported distribution parameter: {min(unexpected)}."
        )
    # The native helper also seeds NumPy's legacy global generator.
    with RANDOM_LOCK:
        state = np.random.get_state()
        try:
            with np.errstate(invalid="raise", over="raise", divide="raise"):
                values = get_random_data(
                    time,
                    seed=settings.seed,
                    random_type=settings.random_type,
                    min_=minimum,
                    range_=span,
                    **settings.parameters,
                )
        except (
            ValueError,
            TypeError,
            FloatingPointError,
            OverflowError,
        ) as error:
            raise ToolError(
                f"Cannot generate a finite synthetic history: {error}"
            ) from error
        finally:
            np.random.set_state(state)
    if not np.isfinite(values).all():
        raise ToolError("Synthetic data must contain only finite values.")
    return values


def synthetic_history(
    source: SyntheticHistory,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the sampled time axis and stress data with native helpers."""
    expected_samples = source.duration * source.fs
    if not 1 < expected_samples <= MAX_SYNTHETIC_SAMPLES:
        raise ToolError("Generate between 2 and 100,000 samples.")
    try:
        time = get_sampled_time(source.fs, source.duration) + source.start
    except (ValueError, OverflowError) as error:
        raise ToolError(f"Invalid synthetic time axis: {error}") from error
    if not 2 <= len(time) <= MAX_SYNTHETIC_SAMPLES:
        raise ToolError("Generate between 2 and 100,000 samples.")
    if not np.isfinite(time).all() or not np.all(time[1:] > time[:-1]):
        raise ToolError("Synthetic sample times must be finite and increasing.")
    return time, scaled_random(time, source, source.minimum, source.maximum)


def synthetic_loads(source: SyntheticLoads) -> list[TensileLoad]:
    """Generate blocks in draw order using a common minimum and duration."""
    if source.lowest_peak <= source.minimum:
        raise ToolError("The lowest peak must exceed the common minimum.")
    peaks = scaled_random(
        np.arange(source.block_count, dtype=float),
        source,
        source.lowest_peak,
        source.highest_peak,
    )
    return [
        TensileLoad(
            minimum=source.minimum,
            maximum=float(peak),
            cycles=source.cycles_per_block,
        )
        for peak in peaks
    ]


def generate_synthetic_data(
    specification: SyntheticInput, export_csv: StrictBool = False
) -> SyntheticSummary:
    """Generate reproducible stress samples or ordered tensile blocks.

    Reuse the returned specification as an analysis input to regenerate
    the full data. The preview contains at most five rows.
    """
    if isinstance(specification, SyntheticHistory):
        time, stress = synthetic_history(specification)
        row_count = len(time)
        preview = [
            SampleRow(time=time[index], stress=stress[index])
            for index in preview_indices(row_count)
        ]
        columns = SAMPLE_COLUMNS
        rows = zip(time, stress)
    else:
        blocks = synthetic_loads(specification)
        row_count = len(blocks)
        preview = [blocks[index] for index in preview_indices(row_count)]
        columns = LOAD_COLUMNS
        rows = (
            (block.minimum, block.maximum, block.cycles) for block in blocks
        )
    export = (
        write_csv_rows("synthetic", columns, rows, row_count)
        if export_csv
        else None
    )
    return SyntheticSummary(
        specification=specification,
        row_count=row_count,
        preview=preview,
        export=export,
    )
