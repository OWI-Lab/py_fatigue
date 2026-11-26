[![version](https://img.shields.io/pypi/v/py_fatigue)](https://pypi.org/project/py-fatigue/)
[![python versions](https://img.shields.io/pypi/pyversions/py_fatigue)](https://pypi.org/project/py-fatigue/)
[![license](https://img.shields.io/github/license/owi-lab/py_fatigue)](https://github.com/OWI-Lab/py_fatigue/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/OWI-Lab/py_fatigue/branch/develop/graph/badge.svg?token=CM4H0C3LVY)](https://codecov.io/gh/OWI-Lab/py_fatigue)
[![pytest](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/ci.yml?label=pytest)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/ci.yml)
[![lint](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/ci.yml?label=lint)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/ci.yml)
[![issues](https://img.shields.io/github/issues/owi-lab/py_fatigue)](https://github.com/OWI-Lab/py_fatigue/issues)
[![CI/CD](https://github.com/OWI-Lab/py_fatigue/actions/workflows/cd.yml/badge.svg)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/cd.yml)
[![documentation](https://github.com/OWI-Lab/py_fatigue/actions/workflows/pages/pages-build-deployment/badge.svg?label=docs)](https://owi-lab.github.io/py_fatigue/)

<!-- Insert the py-fatigue logo here -->
<p align="center">
  <img src="https://owi-lab.github.io/py_fatigue/_static/_img/py-fatigue-logo-with-name.png" alt="py-fatigue logo" width="height"/>
</p>

Py-Fatigue is a Python package for cycle-counting, fatigue analysis and crack propagation prediction. It is developed by [OWI-Lab](https://www.owi-lab.be/) at Vrije Universiteit Brussel. Full documentation can be found [**here**](https://owi-lab.github.io/py_fatigue/).

The main features of Py-Fatigue are:
- a powerful cycle-counting implementation based on the ASTM E1049-85 rainflow method that retrieves the main class of the package: ``CycleCount``
- capability of storing the ``CycleCount`` results in a sparse format for storage and memory efficiency
- easy applicability of multiple mean stress effect correction models
- implementation of low-frequency fatigue recovery when "summing" multiple ``CycleCount`` instances
- fatigue analysis through the combination of SN curves and multiple damage accumulation models
- crack propagation analysis through the combination of the Paris' law and multiple crack geometries
- and more...

Py-Fatigue 3 uses a Rust numerical extension with [``numpy``](https://numpy.org/) and [``pandas``](https://pandas.pydata.org/), for the analytical part, and [``matplotlib``](https://matplotlib.org/) as well as [``pyecharts``](https://pyecharts.org/) for the plotting part.

Examples in marimo offer **Static** (Matplotlib) and **ECharts** views. Pass a Matplotlib figure to `py_fatigue.plotting.to_echarts(fig)` to get an interactive chart, or call `sn_curve.echarts()` for a single S–N curve.

## Installation requirements

> [!IMPORTANT]  
> Py-Fatigue `v1.*.*` is hosted on branch `main` and requires Python [3.8, 3.9, 3.10].
>
> Py-Fatigue `v2.*.*` is hosted on branch `develop` and is compatible with Python [3.10, 3.11, 3.12, 3.13].
>
> Py-Fatigue 3 targets 64-bit CPython 3.10–3.14.
> Installing from source requires Rust. Platform wheels include the native extension.

### Installation

Py-Fatigue can be installed via pip:

```bash
pip install py_fatigue
```

## Example notebooks

Use Python 3.11 or newer for marimo and the documentation. Install Rust and
the development dependencies, then open the notebooks from the repository root.

```bash
uv sync --group dev
uv run marimo edit notebooks
```

Start with `00-development_guide.py` or pick a lesson from the
[numbered notebook index](docs/user/examples/01-notebook_setup.md).

## Build the documentation locally

```bash
uv sync --group dev --group docs
uv run --no-sync invoke docs.build
```

Open `_build/docs/index.html` for the architecture guide, API reference,
and rendered examples.

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
In Github (2.0.1). https://www.owi-lab.be/. https://owi-lab.github.io/py_fatigue
```

### BibTeX-style

```tex
@misc{dantuono-2022,
	author = {given-i=P.D., given=Pietro, family=D'Antuono and given-i=W.W., given=Wout, family=Weijtjens and given-i=C.D., given=Christof, family=Devriendt},
	publisher = {https://www.owi-lab.be/},
	title = {{Py-Fatigue}},
	year = {2022},
	url = {https://owi-lab.github.io/py_fatigue},
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
	url = {https://owi-lab.github.io/py_fatigue},
	version = {2.0.1},
}
```

## License

The package is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Acknowledgements

Py-Fatigue was originally developed in the framework of the [MAXWind project](https://www.owi-lab.be/maxwind/), funded by the Federale Overheidsdienst Economie, KMO, Middenstand en Energie (FOD Economie) in the framework of the Energy Transition Fund (ETF).
