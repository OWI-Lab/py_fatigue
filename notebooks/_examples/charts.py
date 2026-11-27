"""Share the documentation chart style across marimo notebooks."""

from pathlib import Path

import marimo as mo
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure

from py_fatigue.plotting import to_echarts


def configure_plots() -> None:
    """Use the documentation font and transparent Matplotlib surfaces."""
    font = (
        Path(__file__).resolve().parents[2]
        / "docs/stylesheets/fonts/IBMPlexMono-Regular.ttf"
    )
    font_manager.fontManager.addfont(str(font))
    mpl.rcParams.update(
        {
            "font.family": ["IBM Plex Mono", "DejaVu Sans Mono"],
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
        }
    )


_ECHARTS_FONT = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap");
html, body { margin: 0; background: transparent; }
</style>
"""

_ECHARTS_THEME = """
<script>
const scheme = matchMedia("(prefers-color-scheme: dark)");
const notebook = window.frameElement?.ownerDocument;
const originalInks = new WeakMap();
const neutralInk = /^#(?:000000|222222|333333|ffffff)$/i;

function darkTheme() {
  const choice = notebook?.documentElement.dataset.docsTheme;
  if (choice === "light" || choice === "dark") return choice === "dark";
  if (notebook?.body.classList.contains("dark")) return true;
  if (notebook?.body.classList.contains("light")) return false;
  return scheme.matches;
}

function styleCharts() {
  const dark = darkTheme();
  // Matching schemes keep the browser's iframe canvas transparent.
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
  const ink = dark ? "#e2eaf3" : "#18376a";
  const rule = dark ? "#425165" : "#ccd4de";
  const text = { fontFamily: "IBM Plex Mono, monospace", color: ink };
  for (const element of document.querySelectorAll(".chart-container")) {
    const chart = window.echarts?.getInstanceByDom(element);
    if (!chart) continue;
    const options = chart.getOption();
    if (!originalInks.has(chart)) {
      originalInks.set(chart, options.series.map(({lineStyle, itemStyle}) => {
        const colors = {};
        if (neutralInk.test(lineStyle?.color)) colors.lineStyle = lineStyle.color;
        if (neutralInk.test(itemStyle?.color)) colors.itemStyle = itemStyle.color;
        return colors;
      }));
    }
    const series = originalInks.get(chart).map((colors) => Object.fromEntries(
      Object.entries(colors).map(([key, color]) => [
        key, { color: dark || color.toLowerCase() === "#ffffff" ? ink : color },
      ]),
    ));
    const axes = (list) => list.map(() => ({
      axisLabel: text,
      nameTextStyle: text,
      axisLine: { lineStyle: { color: ink } },
      splitLine: { lineStyle: { color: rule } },
    }));
    chart.setOption({
      backgroundColor: "transparent",
      textStyle: text,
      title: { textStyle: text },
      legend: { textStyle: text },
      tooltip: { textStyle: text, backgroundColor: dark ? "#202a36" : "#fcfcfb" },
      toolbox: { iconStyle: { borderColor: ink } },
      visualMap: options.visualMap?.map(() => ({ textStyle: text })),
      xAxis: axes(options.xAxis),
      yAxis: axes(options.yAxis),
      series,
    });
    chart.resize();
  }
}
styleCharts();
document.fonts.load('12px "IBM Plex Mono"').then(styleCharts, styleCharts);
scheme.addEventListener("change", styleCharts);
if (notebook) {
  const observer = new MutationObserver(styleCharts);
  observer.observe(notebook.documentElement, { attributes: true, attributeFilter: ["data-docs-theme"] });
  observer.observe(notebook.body, { attributes: true, attributeFilter: ["class"] });
  window.addEventListener("pagehide", () => observer.disconnect(), { once: true });
}
window.addEventListener("resize", styleCharts);
</script>
"""


def plot_views(figure: Figure, *, height: str | None = None) -> mo.Html:
    """Display the same plot as ECharts and as a transparent static image."""
    page = (
        to_echarts(figure)
        .render_embed()
        .replace(
            "https://assets.pyecharts.org/assets/v6/echarts.min.js",
            "https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js",
        )
    )
    page = page.replace("</head>", _ECHARTS_FONT + "</head>")
    interactive = mo.iframe(
        page.replace("</body>", _ECHARTS_THEME + "</body>"),
        height=height or f"{400 * len(figure.axes) + 20}px",
    ).style({"color-scheme": "var(--notebook-color-scheme, inherit)"})
    static = mo.as_html(figure)
    return mo.ui.tabs(
        {
            "ECharts": interactive,
            "Static": mo.Html(
                '<div class="notebook-static-plot" '
                'style="filter: var(--notebook-plot-filter, none)">'
                f"{static.text}</div>"
            ),
        }
    )


configure_plots()
