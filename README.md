It provides:

- a powerful cycle-counting implementation based on the ASTM E1049-85 rainflow method that retrieves the main class of the package: ``CycleCount``
- capability of storing the ``CycleCount`` results in a sparse format for storage and memory efficiency
- easy applicability of multiple mean stress effect correction models
- implementation of low-frequency fatigue recovery when "summing" multiple ``CycleCount`` instances
- fatigue analysis through the combination of SN curves and multiple damage accumulation models
- crack propagation analysis through the combination of the Paris' law and multiple crack geometries
- and more...

Py-Fatigue is heavily based on [``numba``](https://numba.pydata.org/), [numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/), for the analytical part, and [``matplotlib``](https://matplotlib.org/) for the plotting part.

The package is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).