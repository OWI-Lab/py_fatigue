It provides:

- a powerful cycle-counting implementation based on the ASTM E1049-85 rainflow method that retrieves the main class of the package: ``CycleCount``
- capability of storing the ``CycleCount`` results in a sparse format for storage and memory efficiency
- easy applicability of multiple mean stress effect correction models
- implementation of low-frequency fatigue recovery when "summing" multiple ``CycleCount`` instances
- fatigue analysis through the combination of SN curves and multiple damage accumulation models
- crack propagation analysis through the combination of the Paris' law and multiple crack geometries
- and more...

Py-Fatigue is heavily based on [``numba``](https://numba.pydata.org/), [``numpy``](https://numpy.org/) and [``pandas``](https://pandas.pydata.org/), for the analytical part, and [``matplotlib``](https://matplotlib.org/) for the plotting part.

## To cite Py-Fatigue

If you use Py-Fatigue in your research, please cite the following paper:

### BibTeX-style

```tex
@misc{dantuono-2022,
	author = {given-i=P.D., given=Pietro, family=D'Antuono and given-i=W.W., given=Wout, family=Weijtjens and given-i=C.D., given=Christof, family=Devriendt},
	publisher = {https://www.owi-lab.be/},
	title = {{Py-Fatigue}},
	year = {2022},
	url = {https://pietrodantuono.github.io/py_fatigue},
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
	url = {https://pietrodantuono.github.io/py_fatigue},
	version = {1.0.3},
}
```

### APA 7-style

```
D’Antuono, P. D., Weijtjens, W. W., & Devriendt, C. D. (2022). Py-Fatigue [Software]. In Github (1.0.3). https://www.owi-lab.be/. https://pietrodantuono.github.io/py_fatigue
```

## License

The package is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
