![version](https://img.shields.io/pypi/v/py_fatigue)
![python versions](https://img.shields.io/pypi/pyversions/py_fatigue)
![license](https://img.shields.io/github/license/owi-lab/py_fatigue)
![pytest](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/coverage.yml?label=pytest)
![lint](https://img.shields.io/github/actions/workflow/status/owi-lab/py_fatigue/lint.yml?label=lint)
![issues](https://img.shields.io/github/issues/owi-lab/py_fatigue)
![downloads](https://img.shields.io/github/downloads/owi-lab/py_fatigue/total)
![size](https://img.shields.io/github/repo-size/owi-lab/py_fatigue)

It provides:

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

Py-Fatigue requires Python 3.8 or higher. It is a 64-bit package and it is not compatible with 32-bit Python.

## To cite Py-Fatigue

If you use Py-Fatigue in your research, please cite the following paper:

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

### APA 7-style

```
D’Antuono, P. D., Weijtjens, W. W., & Devriendt, C. D. (2022). Py-Fatigue [Software]. In Github (1.0.3). https://www.owi-lab.be/. https://owi-lab.github.io/py_fatigue
```

## License

The package is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
