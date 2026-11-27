"""Render native Py-Fatigue figures as MCP images or interactive charts."""

from base64 import b64encode
from io import BytesIO
from threading import Lock
from uuid import uuid4

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import EmbeddedResource, ImageContent, TextResourceContents
import numpy as np
import pandas as pd

from py_fatigue import ParisCurve, WalkerCurve
from py_fatigue.crack_growth import NasgroCurve
from py_fatigue.plotting import to_echarts

from .crack_growth import GEOMETRY_TYPES, run_ordered_growth
from .cycle_count import build_cycle_count
from .geometry import PLOT_GEOMETRY_TYPES
from .plot_inputs import (
    CyclePlot,
    GeometryPlot,
    GrowthPlot,
    InlinePlotRequest,
    MinerPlot,
    NasgroPlot,
    PlotFormat,
    PlotRequest,
    SNPlot,
)
from .stress_life import (
    CYCLE_COLUMNS,
    InlineCycles,
    InlineHistory,
    InlineNativeCycles,
    build_sn_curve,
    local_miner_source,
    miner_table,
)

PLOT_LOCK = Lock()
MAX_HTML_BYTES = 2 * 1024 * 1024
PlotContent = ImageContent | EmbeddedResource


def native_figure(
    request: PlotRequest | InlinePlotRequest,
) -> tuple[Figure, Axes]:
    """Select the native plot that corresponds to a validated request."""
    if isinstance(request, CyclePlot):
        count = build_cycle_count(request.stress, request.method)
        if request.view == "residuals":
            return count.plot_residuals_sequence(rainflow_method=request.method)
        if request.view == "half_cycles":
            return count.plot_half_cycles_sequence(
                rainflow_method=request.method
            )
        return count.plot_histogram(
            plot_type=request.axes,
            mode=request.mode,
            bins=request.bins,
            bin_range=request.bin_range,
            normed=request.normalize,
        )
    if isinstance(request, SNPlot):
        if (request.cycles is None) != (request.stress_ranges is None):
            raise ToolError(
                "Supply paired cycles and stress ranges for the overlay."
            )
        return build_sn_curve(request.curve).plot(
            cycles=request.cycles,
            stress_range=request.stress_ranges,
            scale=request.scale,
        )
    if isinstance(request, MinerPlot):
        source = request.source
        if isinstance(
            source, (InlineHistory, InlineCycles, InlineNativeCycles)
        ):
            source = local_miner_source(source)
        table = miner_table(source, request.curve)
        if not len(table):
            raise ToolError("There are no Miner damage rows to plot.")
        frame = pd.DataFrame(table, columns=[*CYCLE_COLUMNS, "pm_damage"])
        # The native accessor needs the cycle table columns and units.
        frame["mean_stress"] = np.nan
        frame._metadata = {"unit": "MPa"}
        if request.view == "histogram":
            return frame.miner.plot_histogram()
        return frame.miner.plot()
    if isinstance(request, GeometryPlot):
        spec = request.geometry
        geometry = PLOT_GEOMETRY_TYPES[spec.case](
            **spec.model_dump(exclude={"case"})
        )
        return geometry.plot()
    if isinstance(request, GrowthPlot):
        result = run_ordered_growth(
            request.geometry, request.curve, request.loads, request.controls
        )
        if request.view == "profile":
            spec = request.geometry
            geometry = GEOMETRY_TYPES[spec.case](
                **spec.model_dump(exclude={"case"})
            )
            return geometry.plot(result=result)
        return result.plot()
    if isinstance(request, NasgroPlot):
        return NasgroCurve(**request.curve.model_dump()).plot(
            length=request.length,
            k_min=request.k_min,
            k_max=request.k_max,
            scale=request.scale,
        )
    if (request.sif is None) != (request.growth_rate is None):
        raise ToolError(
            "Supply paired stress intensities and growth rates for the overlay."
        )
    if np.size(request.slope) != np.size(request.intercept):
        raise ToolError("Supply one Paris or Walker coefficient per slope.")
    growth_curve = WalkerCurve if request.kind == "walker_curve" else ParisCurve
    curve = growth_curve(
        slope=request.slope,
        intercept=request.intercept,
        threshold=request.threshold,
        critical=np.inf if request.critical is None else request.critical,
        load_ratio=request.load_ratio,
        walker_exponent=request.walker_exponent,
    )
    return curve.plot(
        sif=request.sif, growth_rate=request.growth_rate, scale=request.scale
    )


def render_native_plot(
    request: PlotRequest | InlinePlotRequest, format: PlotFormat
) -> PlotContent:
    """Return a PNG or an HTML resource from the same native figure."""
    with PLOT_LOCK:
        if plt.get_backend().lower() != "agg":
            plt.switch_backend("Agg")
        open_figures = set(plt.get_fignums())
        try:
            figure, _ = native_figure(request)
            if format == "echarts":
                html = to_echarts(figure).render_embed()
                if len(html.encode("utf-8")) > MAX_HTML_BYTES:
                    raise ToolError(
                        "Interactive chart exceeds 2 MiB. Request PNG or use fewer inputs."
                    )
                return EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"py-fatigue://plots/{request.kind}-{uuid4().hex}.html",
                        mime_type="text/html",
                        text=html,
                    ),
                )
            with BytesIO() as buffer:
                figure.savefig(buffer, format="png", dpi=120)
                data = b64encode(buffer.getvalue()).decode("ascii")
            return ImageContent(type="image", data=data, mime_type="image/png")
        except (
            ValueError,
            TypeError,
            AssertionError,
            RuntimeError,
            OverflowError,
        ) as error:
            raise ToolError(
                f"Cannot render a native Py-Fatigue plot: {error}"
            ) from error
        finally:
            for number in set(plt.get_fignums()) - open_figures:
                plt.close(number)


def plot_py_fatigue(
    request: PlotRequest, format: PlotFormat = "png"
) -> PlotContent:
    """Use Py-Fatigue's native plots before writing custom plotting code.

    Choose a CycleCount histogram, residuals or half-cycles, an S–N,
    Paris, Walker or NASGRO curve, Miner damage, a geometry sketch, or
    ordered crack-growth history or profile. Select PNG for an MCP image
    or ECharts for an embedded HTML resource.
    """
    return render_native_plot(request, format)
