[![version](https://img.shields.io/pypi/v/py_fatigue)](https://pypi.org/project/py-fatigue/)
[![python versions](https://img.shields.io/pypi/pyversions/py_fatigue)](https://pypi.org/project/py-fatigue/)
[![license](https://img.shields.io/github/license/owi-lab/py_fatigue)](https://github.com/OWI-Lab/py_fatigue/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/OWI-Lab/py_fatigue/branch/develop/graph/badge.svg?token=CM4H0C3LVY)](https://codecov.io/gh/OWI-Lab/py_fatigue)
[![pytest](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/ci.yml?label=pytest)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/ci.yml)
[![lint](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/ci.yml?label=lint)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/ci.yml)
[![issues](https://img.shields.io/github/issues/owi-lab/py_fatigue)](https://github.com/OWI-Lab/py_fatigue/issues)
[![CI/CD](https://github.com/OWI-Lab/py_fatigue/actions/workflows/cd.yml/badge.svg)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/cd.yml)
[![documentation](https://github.com/OWI-Lab/py_fatigue/actions/workflows/pages/pages-build-deployment/badge.svg?label=docs)](https://py-fatigue.dev/)

![Py-Fatigue](https://raw.githubusercontent.com/OWI-Lab/py_fatigue/develop/docs/_static/_img/py-fatigue-readme-logo.svg)

Py-Fatigue is a Python package for stress–life fatigue, crack-growth analysis,
and cycle counting. It is developed at Vrije Universiteit Brussel by
[OWI-Lab](https://www.owi-lab.be/).

[Stable documentation](https://py-fatigue.dev/) · [Migrate from 2.x](https://py-fatigue.dev/3.0.0/user/migrate-v3.html)

## What you can analyze

- **Stress–life fatigue** with S–N curves, mean-stress corrections, and
  linear or nonlinear damage accumulation. Estimate fatigue life and
  calculate damage-equivalent stresses and moments.
  [Stress–life example](https://py-fatigue.dev/3.0.0/user/examples/06-stress-life-damage.html)
- **Crack growth** with Paris and Walker curves or ordered NASGRO load
  blocks. Evaluate stress intensity for supported geometries and plot the
  initial and final crack profiles.
  [Crack-growth example](https://py-fatigue.dev/3.0.0/user/examples/07-ordered-crack-growth.html)
- **Cycle counting** with ASTM E1049-85 rainflow counting and `CycleCount`.
  Import SEP005 or pyExSi histories, store sparse rainflow histograms, and
  recover low-frequency cycles when combining counting windows.
  [Cycle-counting example](https://py-fatigue.dev/3.0.0/user/examples/03-count-stress-cycles.html)

Py-Fatigue 3 uses Rust numerical kernels with NumPy and pandas. Plot material
curves, geometries, and analysis results with Matplotlib or ECharts.

Examples in marimo offer **Static** (Matplotlib) and **ECharts** views. Pass a Matplotlib figure to `py_fatigue.plotting.to_echarts(fig)` to get an interactive chart, or call `sn_curve.echarts()` for a single S–N curve.

## Installation

Install the stable release with:

```bash
pip install py-fatigue
```

Py-Fatigue 3 supports 64-bit CPython 3.10–3.15. Platform wheels include the
Rust extension. Source builds require Rust.

> [!NOTE]
> The `main` branch is dedicated to Py-Fatigue 1, which has its own
> development lifecycle and supports Python 3.8–3.10. Py-Fatigue 2 and 3
> are maintained on `develop`. Py-Fatigue 2 supports Python 3.10–3.14.

## MCP server

Use Py-Fatigue from an MCP client to count, combine, and scale cycles,
calculate Miner damage, model crack growth, and plot results. The
[MCP guide](https://py-fatigue.dev/mcp/01-overview.html) covers local and
hosted setup, including Zed, and provides examples.

## Example notebooks

Use Python 3.11 or newer for marimo and the documentation. Install Rust and
the development dependencies, then open the notebooks from the repository root.

```bash
uv sync --group dev
uv run marimo edit notebooks
```

Start with `00-development_guide.py` or pick a lesson from the
[numbered notebook index](https://py-fatigue.dev/3.0.0/user/examples/08-example-notebooks.html).
The end-to-end tutorials are in [notebooks/tutorials](https://github.com/OWI-Lab/py_fatigue/tree/develop/notebooks/tutorials).

## Build the documentation locally

```bash
uv sync --group dev --group docs
uv run --no-sync invoke docs.build
```

Open `_build/docs/index.html` for the architecture guide, API reference,
and rendered examples.

### Publish a documentation version

Run the **Documentation** workflow from `develop` and pass the existing
release tag. Each tag is built from its own source and published to its
versioned URL. For example, `v3.0.0` uses `/3.0.0/`.

```bash
gh workflow run pages.yml --ref develop -f publish=true -f release_tag=v3.0.0 -f stable=true
```

For a documentation refresh, merge the updates into `develop` and set
`refresh=true`. The tag must match the version declared on `develop`.

```bash
gh workflow run pages.yml --ref develop -f publish=true -f release_tag=v3.0.0 -f refresh=true -f stable=true
```

Use `stable=true` only when a final release should also be served at the
site root. A documentation publication keeps the other versioned sites.

## Upgrading from 2.x

Existing fatigue classes, pandas accessors and cycle-count operations retain
their import paths. Use `py_fatigue.stress_life` and `py_fatigue.crack_growth`
for domain-specific entry points.

The Rust extension loads without a JIT warmup. Remove calls to `warmup_numba`.
Use ordinary mappings for geometry inputs and `py_bisect` or `py_newton` for
Python callback solvers. The Numba dictionary and dispatcher helpers, compiled
residual alias and JIT settings have been retired.

## To cite Py-Fatigue

If you use Py-Fatigue in your research, please use the following citation:

### APA 7-style

```
D’Antuono, P. D., Weijtjens, W. W., & Devriendt, C. D. (2022). Py-Fatigue [Software].
In Github (3.0.1). https://www.owi-lab.be/. https://py-fatigue.dev
```

### BibTeX-style

```tex
@misc{dantuono-2022,
	author = {given-i=P.D., given=Pietro, family=D'Antuono and given-i=W.W., given=Wout, family=Weijtjens and given-i=C.D., given=Christof, family=Devriendt},
	publisher = {https://www.owi-lab.be/},
	title = {{Py-Fatigue}},
	year = {2022},
	url = {https://py-fatigue.dev},
}
```

### BibLaTeX-style

```tex
@software{dantuono-2022,
	author = {given-i=P.D., given=Pietro, family=D'Antuono and given-i=W.W., given=Wout, family=Weijtjens and given-i=C.D., given=Christof, family=Devriendt},
	date = {2022},
	language = {english},
	publisher = {https://www.owi-lab.be/},
	title = {Py-Fatigue},
	type = {software},
	url = {https://py-fatigue.dev},
	version = {3.0.1},
}
```

## License

The package is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
The source distribution and wheels include the
[Rust dependency notices](https://github.com/OWI-Lab/py_fatigue/blob/develop/THIRD_PARTY_NOTICES).

## Acknowledgements

Py-Fatigue was originally developed in the framework of the [MAXWind project](https://www.owi-lab.be/maxwind/), funded by the Federale Overheidsdienst Economie, KMO, Middenstand en Energie (FOD Economie) in the framework of the Energy Transition Fund (ETF).
