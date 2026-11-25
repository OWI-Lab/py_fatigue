# 1. Set up a notebook running *py-fatigue* {#1. Py-fatigue notebook setup}

From the repository root, install the development environment and open the development guide in marimo.

``` bash
uv sync --group dev
uv run marimo edit notebooks/development_guide.py
```

The project is installed in editable mode. Python cells can import py_fatigue directly. The notebook includes examples of ASTM and four-point rainflow counting and S--N curve creation.

Run the notebook as a script to check that its cells execute headlessly.

``` bash
uv run python notebooks/development_guide.py
```

[View the rendered development notebook](../../development_guide.html).
