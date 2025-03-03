[![version](https://img.shields.io/pypi/v/py_fatigue)](https://pypi.org/project/py-fatigue/)
[![python versions](https://img.shields.io/pypi/pyversions/py_fatigue)](https://pypi.org/project/py-fatigue/)
[![license](https://img.shields.io/github/license/owi-lab/py_fatigue)](https://github.com/OWI-Lab/py_fatigue/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/OWI-Lab/py_fatigue/branch/main/graph/badge.svg?token=CM4H0C3LVY)](https://codecov.io/gh/OWI-Lab/py_fatigue)
[![pytest](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/ci.yml?label=pytest)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/ci.yml)
[![lint](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/ci.yml?label=lint)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/ci.yml)
[![issues](https://img.shields.io/github/issues/owi-lab/py_fatigue)](https://github.com/OWI-Lab/py_fatigue/issues)
[![CI/CD](https://github.com/OWI-Lab/py_fatigue/actions/workflows/cd.yml/badge.svg)](https://github.com/OWI-Lab/py_fatigue/actions/workflows/cd.yml)
[![documentation](https://github.com/OWI-Lab/py_fatigue/actions/workflows/pages/pages-build-deployment/badge.svg?label=docs)](https://owi-lab.github.io/py_fatigue/)
[![Binder Tutorials](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OWI-Lab/py-fatigue-tutorials/HEAD)

<!-- Insert the py-fatigue logo here -->
<p align="center">
  <img src="https://owi-lab.github.io/py_fatigue/_static/py-fatigue-logo-with-name.png" alt="py-fatigue logo" width="height"/>
</p>

Py-Fatigue is a Python package for cycle-conuting, fatigue analysis and crack propagation prediction. It is developed by [OWI-Lab](https://www.owi-lab.be/) at Vrije Universiteit Brussel. Full documentation can be found [**here**](https://owi-lab.github.io/py_fatigue/).

- a powerful cycle-counting implementation based on the ASTM E1049-85 rainflow method that retrieves the main class of the package: ``CycleCount``
- capability of storing the ``CycleCount`` results in a sparse format for storage and memory efficiency
- easy applicability of multiple mean stress effect correction models
- implementation of low-frequency fatigue recovery when "summing" multiple ``CycleCount`` instances
- fatigue analysis through the combination of SN curves and multiple damage accumulation models
- crack propagation analysis through the combination of the Paris' law and multiple crack geometries
- and more...

Py-Fatigue is heavily based on [``numba``](https://numba.pydata.org/), [``numpy``](https://numpy.org/) and [``pandas``](https://pandas.pydata.org/), for the analytical part, and [``matplotlib``](https://matplotlib.org/) as well as [``plotly``](https://plotly.com/python/) for the plotting part.

Therefore, it is highly recommended to have a look at the documentation of these packages as well.

## Installation requirements

Py-Fatigue v1.*.* requires Python [3.8, 3.9, 3.10], while py-fatigue v2.*.* is compatible with Python [3.10, 3.11, 3.12, 3.13]. It is a 64-bit package, hence not compatible with 32-bit Python.

### Installation

Py-Fatigue can be installed via pip:

```bash
pip install py_fatigue
```

## To cite Py-Fatigue

If you use Py-Fatigue in your research, please use the following citation:

### APA 7-style

```
D’Antuono, P. D., Weijtjens, W. W., & Devriendt, C. D. (2022). Py-Fatigue [Software]. In Github (1.0.3). https://www.owi-lab.be/. https://owi-lab.github.io/py_fatigue
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
	version = {1.0.3},
}
```

## License

The package is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Acknowledgements

Py-Fatigue was originally developed in the framework of the [MAXWind project](https://www.owi-lab.be/maxwind/), funded by the Federale Overheidsdienst Economie, KMO, Middenstand en Energie (FOD Economie) in the framework of the Energy Transition Fund (ETF).
