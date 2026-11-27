# 8. Run the example notebooks { #run-the-example-notebooks }

From the repository root, use Python 3.11 or newer and have Rust's `cargo` on
your PATH.

```bash
uv sync --group dev --group docs
uv run --no-sync marimo edit notebooks
```

The launch command uses the project's `.venv` and the local
`py_fatigue` checkout. An editor started with `uvx marimo` uses a
separate environment. If a notebook reports
`ModuleNotFoundError: No module named 'py_fatigue.plotting'`, open it
in an editor launched with the command above. Use **Run all cells** if
marimo restores the previous error from saved output.

## Edit the Getting Started notebook

The [Getting Started guide](../01-absolute-noob.md#runnable-notebook) is a
notebook with saved outputs. The documentation build generates it from
`notebooks/guides/getting-started.md`. Each worked guide also embeds
the notebook for its analysis, with outputs ready to inspect.

Generate and open the source in marimo to edit or rerun its cells.

```bash
uv run --no-sync invoke docs.getting-started
uv run --no-sync marimo edit _build/docs/user/getting-started-notebook.py
```

The embedded copy displays the saved outputs. The local editor runs
Python with the project's installed Rust extension.

## Example catalogue

The plots have **ECharts** and **Static** tabs. Choose ECharts for hover
values, zooming and image export, or Static for Matplotlib. The
ECharts tab loads Apache ECharts from jsDelivr.

The first digit groups the topic. The second gives the lesson order within
that topic. Start at `00` for a tour, then work through cycle counting
(`1x`), stress-life analysis (`2x`), and crack growth (`3x`).

Every lesson applies its workflow to the same **300,000 seeded tensile
cycles**. The shared notebook helper uses `py_fatigue.testing.get_random_data`
to generate the peaks. Rainflow examples count the full **600,001-sample**
waveform formed by returning to a fixed tensile minimum between peaks.
History plots sample points for display after the full calculation.

| Lesson | What it teaches |
| --- | --- |
| [00 Development guide](../../00-development_guide.html) | Compare ASTM and four-point residuals and overlay S–N curves. |
| [10 Rainflow counting](../../10-rainflow_counting.html) | Count full and half cycles, export a sparse spectrum, and compare it after reloading. |
| [11 Cycle count aggregation](../../11-cycle_count_aggregation.html) | Add adjacent windows, recover residuals, and compare the spectrum and Miner damage with a whole-signal count. |
| [20 S–N curves and Miner damage](../../20-sn_curves_and_miner_damage.html) | Build a curve from knee points and see the effect of the endurance limit on life and damage. |
| [21 Mean stress and equivalent loads](../../21-mean_stress_and_equivalent_loads.html) | Compare corrections, Miner damage, damage equivalent stress, and damage equivalent moment. |
| [22 Nonlinear damage](../../22-nonlinear_damage.html) | Compare Miner and Pavlou under reversed load orders and plot a Theil effective stress history. |
| [30 Legacy Paris growth](../../30-legacy_paris_growth.html) | Compare Paris and Walker curves, then propagate an infinite-surface crack and an external cylinder crack. |
| [31 Plate stress intensity](../../31-plate_stress_intensity.html) | Sketch two plate geometries and compare stress intensities as crack depth changes. |
| [32 NASGRO growth rate](../../32-nasgro_growth_rate.html) | Sweep intensity up to toughness and compare two fixed stress ratios. |
| [33 Ordered crack growth](../../33-ordered_crack_growth.html) | Inspect applied stresses, crack dimensions, and peak stress intensities on a shared cycle range. |
| [34 Synthetic signal to failure](../../34-synthetic_signal_to_failure.html) | Generate a long waveform and follow its ordered cycles to the Miner and crack-growth limits. |
| [35 Through-crack growth](../../35-through_crack_growth.html) | Propagate center and edge plate cracks under the same loads and compare their initial and final profiles. |
| [36 Surface-crack growth](../../36-surface_crack_growth.html) | Compare a semi-elliptical plate crack with an SC07 round-bar crack under the same tensile blocks. Plot their depth histories and crack-front profiles. |
| [37 AFGROW MERC-3 crack-growth sensitivity](../../37-afgrow_merc3_growth_sensitivity.html) | Compare the native Paris solver with a closed-form life and reconstruct the handbook longeron life from digitized geometry factors. |
| [38 AFGROW FAC-1 stress-intensity history](../../38-afgrow_fac1_sif_history.html) | Reconstruct the combined stress intensity for bypass tension and pin bearing and compare published finite-element and NASGRO results. |

The examples draw on workflows in the original OWI-Lab tutorials. They use
self-contained inputs so they can run from this checkout. [OWI-Lab tutorial notebooks](https://github.com/OWI-Lab/py-fatigue-tutorials)

## End-to-end tutorials

Each tutorial defines a material curve, a geometry and a load history, then
runs the analysis and plots the results. Crack-growth tutorials show the
initial and final crack profiles. Each analysis includes a seeded
300,000-cycle load spectrum and explains the outcome.

| Tutorial | Analysis |
| --- | --- |
| [10 Tube in bending](../../tutorials/10-tubular_stress_life.html) | Convert bending moments to stress, aggregate rainflow counts, and compare Miner damage and equivalent loads with an S–N curve. |
| [20 External pipe crack](../../tutorials/20-pipe_paris_growth.html) | Convert axial force to stress and propagate a crack in a hollow cylinder with the Paris law. |
| [30 Through and surface cracks in a plate](../../tutorials/30-ordered_plate_crack_growth.html) | Compare the crack dimensions, stress intensities, and stop conditions under the same ordered loads. |
| [40 AFGROW MERC-3 longeron](../../tutorials/40-afgrow_merc3_longeron.html) | Compare handbook lives, independent integration and Py-Fatigue for three geometry-factor curves. |
| [41 AFGROW FAC-1 loaded hole](../../tutorials/41-afgrow_fac1_loaded_hole.html) | Check Py-Fatigue against the handbook stress intensities and independently integrate an illustrative crack-growth history. |
| [42 AFGROW blend-out repair](../../tutorials/42-afgrow_blend_out_repair.html) | Compare the life of a repaired edge crack against the handbook's reported and printed calculations. |

Open the interactive tutorials from the repository root.

```bash
uv run --no-sync marimo edit notebooks/tutorials
```

Build the local documentation and HTML versions of all notebooks together.

```bash
uv run --no-sync invoke docs.build
```

Open `_build/docs/index.html` to browse the result.
