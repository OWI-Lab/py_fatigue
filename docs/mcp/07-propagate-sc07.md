# 7. Propagate an SC07 round-bar crack

Follow a circular surface crack in a solid 100 mm diameter bar under
three blocks of axial tension. The initial crack depth is 4 mm. The
material curve and loads are synthetic and match the
[SC07 notebook](../36-surface_crack_growth.html).

Start with a connected [local](02-run-locally.md) or [hosted](03-connect-hosted.md)
MCP client. Use remote tensile stress in MPa, lengths in mm, and
stress intensity in MPa·√mm. SC07 advances the deepest point of the
circular crack front.

## Apply the ordered loads

In a prompt-based client, paste the JSON and ask it to call
`ordered_crack_growth` with those exact arguments. Ask for the termination
reason, final cycle count, and final depth in mm. You can also send the
arguments directly.

```json
{
  "geometry": {"case": "SC07", "diameter": 100, "depth": 4},
  "curve": {
    "coefficient": 1e-10, "exponent": 2, "p": 0.5, "q": 1,
    "threshold_zero": 25, "threshold_fanning": 0, "toughness": 500,
    "constraint": 2, "peak_to_flow": 0.3
  },
  "loads": [
    {"minimum": 10, "maximum": 70, "cycles": 50000},
    {"minimum": 20, "maximum": 90, "cycles": 25005},
    {"minimum": 10, "maximum": 60, "cycles": 25000}
  ],
  "controls": {
    "max_cycle_step": 10, "length_tolerance": 1e-6,
    "cycle_tolerance": 1e-4, "max_trials": 20002
  }
}
```

The solver should return **`spectrum_exhausted`** after **100,005 cycles**,
with `dimension_names` equal to `["Depth"]` and `final_state.sizes[0]`
of about **4.180408 mm**. The last accepted state
is in load block index 2, the third block. The stop reason means the
solver completed the supplied load history.

Open the [surface-crack notebook](../36-surface_crack_growth.html) and
find **SC07 under the same ordered loads** to plot the round-bar cross
section and crack depth through the three blocks. To edit and rerun it
from a checkout, run the following command from the repository root.

```sh
uv run --group docs marimo edit notebooks/36-surface_crack_growth.py
```

See [supported geometries and loading limits](../api/damage/ordered_crack_growth.md)
and the [MCP tool reference](08-tool-reference.md#simulate-ordered-crack-growth)
for the full input and output conventions.
