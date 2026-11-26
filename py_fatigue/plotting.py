"""Interactive rendering for Py-Fatigue's Matplotlib plots."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from pyecharts import options as opts
from pyecharts.charts import Line, Page
from pyecharts.commons.utils import JsCode
from pyecharts.globals import Locale

_CurveScale = Literal["loglog", "semilogx", "semilogy"]


def _plot_curve(
    ax: Axes, x: ArrayLike, y: ArrayLike, scale: _CurveScale, **kwargs: Any
) -> None:
    """Draw a curve and set both axes so overlays use the requested scales."""
    if scale == "loglog":
        ax.loglog(x, y, **kwargs)
    elif scale == "semilogx":
        ax.set_yscale("linear")
        ax.semilogx(x, y, **kwargs)
    elif scale == "semilogy":
        ax.set_xscale("linear")
        ax.semilogy(x, y, **kwargs)
    else:
        raise ValueError("scale must be 'loglog', 'semilogx', or 'semilogy'.")


class PlotMixin:
    """Give single-panel plot methods an ECharts counterpart."""

    if TYPE_CHECKING:
        plot: Callable[..., tuple[Figure, Axes]]

    def echarts(self, *args: Any, **kwargs: Any) -> Line:
        """Return a pyecharts chart using the same arguments as `plot()`."""
        open_figures = set(plt.get_fignums())
        fig, ax = self.plot(*args, **kwargs)
        try:
            return to_echarts(ax)
        finally:
            if getattr(fig, "number", None) not in open_figures:
                plt.close(fig)


@overload
def to_echarts(plot: Axes) -> Line:  # noqa: E704
    ...


@overload
def to_echarts(plot: Figure) -> Line | Page:  # noqa: E704
    ...


def to_echarts(plot: Figure | Axes) -> Line | Page:
    """Render a Py-Fatigue plot as an interactive pyecharts chart.

    Built-in chart controls use English.

    Parameters
    ----------
    plot : matplotlib.figure.Figure or matplotlib.axes.Axes
        Figure or axes containing Py-Fatigue lines and/or scatter points.

    Returns
    -------
    pyecharts.charts.Line or pyecharts.charts.Page
        A chart for one panel or a page for multiple panels. Use
        `render()` to save HTML or `render_embed()` to embed it.

    Examples
    --------
    >>> from py_fatigue import SNCurve
    >>> from py_fatigue.plotting import to_echarts
    >>> fig, ax = SNCurve(slope=3.0, intercept=12.0).plot()
    >>> chart = to_echarts(fig)
    >>> chart.options["xAxis"][0]["type"]
    'log'
    """
    if isinstance(plot, Axes):
        return _axes_chart(plot)

    colorbars = {
        collection.colorbar.ax
        for ax in plot.axes
        for collection in ax.collections
        if collection.colorbar is not None
    }
    charts = [_axes_chart(ax) for ax in plot.axes if ax not in colorbars]
    return charts[0] if len(charts) == 1 else Page().add(*charts)


def _axes_chart(ax: Axes) -> Line:
    """Convert a numerical line and scatter panel."""
    chart = Line(init_opts=opts.InitOpts(width="100%", height="400px"))
    chart.locale = Locale.EN
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    tick_format = JsCode("""function (value) {
            const magnitude = Math.abs(value);
            return magnitude > 0 && (magnitude < 0.001 || magnitude >= 1e6)
                ? value.toExponential(2)
                : Number(value.toPrecision(6)).toString();
        }""")
    for name, axis, limits in (
        ("xAxis", ax.xaxis, x_limits),
        ("yAxis", ax.yaxis, y_limits),
    ):
        scale = axis.get_scale()
        if scale not in ("linear", "log"):
            raise ValueError(
                f"ECharts rendering does not support {scale!r} axes"
            )
        axis_opts = chart.options[name][0]
        axis_opts.update(
            type="log" if scale == "log" else "value",
            name=axis.get_label_text(),
            show=axis.get_visible(),
            nameLocation="middle",
            nameGap=36 if name == "xAxis" else 58,
            min=min(limits),
            max=max(limits),
            inverse=bool(limits[0] > limits[1]),
            axisLabel={"hideOverlap": True, "formatter": tick_format},
        )

    chart.options.update(
        animation=False,
        aria={"enabled": True},
        title={"text": ax.get_title(), "left": "center"},
        grid={"top": 58, "bottom": 80, "left": 105, "right": 38},
        toolbox={
            "feature": {"dataZoom": {}, "restore": {}, "saveAsImage": {}},
        },
    )
    series = chart.options["series"]
    legend = []
    for line in ax.lines:
        if not line.get_visible():
            continue
        name = line.get_label()
        if name and not name.startswith("_"):
            legend.append(name)
        color = to_hex(line.get_color())
        linestyle = line.get_linestyle()
        marker = line.get_marker()
        symbol = {
            "D": "diamond",
            "d": "diamond",
            "s": "rect",
            "^": "triangle",
        }.get(marker, "circle")
        x, y = line.get_xydata().T
        if line.get_transform() == ax.get_xaxis_transform():
            y = np.asarray(y_limits)
        elif line.get_transform() == ax.get_yaxis_transform():
            x = np.asarray(x_limits)
        series.append(
            {
                "name": name if name in legend else f"Series {len(series) + 1}",
                "type": "line",
                "data": _points(x, y, ax),
                "showSymbol": str(marker).strip().lower() not in ("", "none"),
                "symbol": symbol,
                "z": line.get_zorder(),
                "symbolSize": line.get_markersize(),
                "itemStyle": {"color": color},
                "lineStyle": {
                    "color": color,
                    "width": 0 if linestyle == "None" else line.get_linewidth(),
                    "type": {"--": "dashed", ":": "dotted", "-.": "dashed"}.get(
                        linestyle, "solid"
                    ),
                    "opacity": line.get_alpha(),
                },
            }
        )

    for collection in ax.collections:
        if (
            not isinstance(collection, PathCollection)
            or not collection.get_visible()
        ):
            continue
        name = collection.get_label()
        if name and not name.startswith("_"):
            legend.append(name)
        x, y = (
            np.ma.asarray(collection.get_offsets(), dtype=float)
            .filled(np.nan)
            .T
        )
        values = collection.get_array()
        scatter = {
            "name": name if name in legend else f"Series {len(series) + 1}",
            "type": "scatter",
            "data": _points(x, y, ax),
            "symbolSize": float(np.sqrt(collection.get_sizes()[0])),
            "z": collection.get_zorder(),
        }
        if values is None:
            colors = collection.get_facecolors()
            if len(colors):
                scatter["itemStyle"] = {"color": to_hex(colors[0])}
        else:
            values = np.ma.asarray(values, dtype=float).filled(np.nan)
            for point, value in zip(scatter["data"], values):
                point.append(float(value) if np.isfinite(value) else None)
            _set_color_scale(chart, collection, len(series), scatter)
        series.append(scatter)

    chart.options["legend"] = {
        "data": list(dict.fromkeys(legend)),
        "type": "scroll",
        "bottom": 0,
    }
    if ax.get_aspect() == 1:
        _keep_equal_aspect(chart)
    chart.add_js_funcs(
        f"new ResizeObserver(() => chart_{chart.chart_id}.resize())"
        f".observe(document.getElementById('{chart.chart_id}'));"
    )
    return chart


def _points(x: np.ndarray, y: np.ndarray, ax: Axes) -> list[list]:
    """Use nulls for gaps, including out-of-domain log values."""
    coordinates = np.column_stack((x, y))
    valid = np.isfinite(coordinates)
    if ax.get_xscale() == "log":
        valid[:, 0] &= x > 0
    if ax.get_yscale() == "log":
        valid[:, 1] &= y > 0
    return np.where(valid, coordinates, None).tolist()


def _set_color_scale(
    chart: Line, collection: PathCollection, index: int, scatter: dict
) -> None:
    """Use the same color range and dimension as the Matplotlib scatter."""
    collection.autoscale_None()
    lower, upper = collection.get_clim()
    if lower is None or upper is None:
        return
    colorbar = collection.colorbar
    label = "Value"
    if colorbar is not None:
        label = colorbar.ax.get_ylabel() or colorbar.ax.get_xlabel() or label
    scatter["dimensions"] = ["x", "y", label]
    scatter["encode"] = {"x": 0, "y": 1, "tooltip": [0, 1, 2]}
    chart.options.setdefault("visualMap", []).append(
        {
            "type": "continuous",
            "min": lower,
            "max": upper,
            "seriesIndex": index,
            "dimension": 2,
            "orient": "horizontal",
            "left": "center",
            "bottom": 25,
            "text": [label, ""],
            "inRange": {
                "color": [
                    to_hex(collection.to_rgba(value))
                    for value in np.linspace(lower, upper, 9)
                ]
            },
        }
    )


def _keep_equal_aspect(chart: Line) -> None:
    """Keep equal data units on the cylinder and stress histogram axes."""
    chart_var = f"chart_{chart.chart_id}"
    chart.add_js_funcs(f"""
        new ResizeObserver(() => {{
            const chart = {chart_var};
            const option = chart.getOption();
            const [x, y] = [option.xAxis[0], option.yAxis[0]];
            const scale = Math.min(
                Math.max(1, chart.getWidth() - 143) / (x.max - x.min),
                250 / (y.max - y.min)
            );
            chart.setOption({{grid: {{
                width: scale * (x.max - x.min), height: scale * (y.max - y.min)
            }}}});
        }}).observe(document.getElementById('{chart.chart_id}'));
        """)
