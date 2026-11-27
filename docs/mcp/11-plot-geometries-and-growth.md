# 11. Plot geometries and crack growth

Use `plot_py_fatigue` to sketch an initial geometry or visualize a
native crack-growth result. All ten ordered cases support native
stress-intensity evaluation with `geometry_stress_intensity`. Their
opening stress and crack-front order depend on the case.

Use mm for dimensions and MPa for stresses. These example geometries
illustrate the inputs. Check the
[case-specific validity limits](../api/damage/ordered_crack_growth.md#what-each-new-name-does)
for your own geometry.

## Through-crack geometries

| Case | Prompt |
| --- | --- |
| TC01 | “Plot the plate with a centered through crack. Use a plate half-width of 50 mm and a crack half-length of 5 mm.” |
| TC02 | “Plot a 50 mm-wide plate with a 5 mm single edge crack.” |
| TC06 | “Plot the through crack in a spherical shell with mean radius 100 mm, thickness 1 mm, and crack half-length 10 mm.” |
| TC07 | “Plot the axial through crack in a cylindrical shell with mean radius 100 mm, thickness 1 mm, and crack half-length 10 mm. Evaluate intensity at 30 and 60 MPa hoop membrane stress.” |
| TC08 | “Plot the circumferential through crack with mean radius 100 mm, wall thickness 1 mm, crack half-length 10 mm, axial length 2500 mm, and Poisson ratio 0.3.” |

## Surface-crack geometries

| Case | Prompt |
| --- | --- |
| SC01 | “Plot a semi-elliptical surface crack with plate half-width 50 mm, thickness 15 mm, depth 6 mm, and surface half-length 10 mm. Report the two tip intensities at 70 MPa membrane stress.” |
| SC05 | “Plot the external circumferential pipe crack with inner radius 20 mm, wall thickness 10 mm, depth 4 mm, and outer arc half-length 5 mm.” |
| SC07 | “Plot a circular surface crack with depth 4 mm in a 100 mm diameter solid bar. Evaluate the deepest-point intensity at 50 and 65 MPa.” |
| SC08 | “Plot the rolled-thread crack with major diameter 12 mm, minor diameter 10 mm, and crack depth 1.5 mm.” |
| SC11 | “Plot two symmetric cracks at a plate hole. Use width 80 mm, thickness 12 mm, hole diameter 12 mm, depth 2.2 mm, and half-length 2.5 mm.” |

## Legacy geometries and tabulated factors

These plot requests use the legacy geometry classes. The ten cases above
are the inputs to the ordered NASGRO solver.

- “Draw the infinite-surface crack with initial depth 3 mm.”
- “Draw a solid cylinder with diameter 100 mm, height 150 mm, and
  initial crack depth 3 mm.”
- “Plot a hollow cylinder with an external 3 mm initial flaw,
  outer diameter 100 mm, wall thickness 15 mm, and height 150 mm.”
- “Use the same hollow cylinder with an internal flaw. Return an
  interactive sketch.”
- “Plot a tabulated factor with initial depth 3 mm, depths
  `[2, 4, 7]` mm, and factors `[1.0, 1.1, 1.4]`.”

## Plot a solved crack history

Start with the [SC07 worked example](07-propagate-sc07.md).
Reuse its `geometry`, `curve`, `loads`, and `controls` in
`plot_py_fatigue` with `request.kind: "crack_growth"`.

- “Show crack depth versus cumulative cycles for the solved SC07 case.”
- “Draw the initial and last accepted SC07 fronts on the same bar cross
  section. Report the final depth and termination reason from the solver.”
- “Return the native interactive history for the same load sequence.”
- “Reverse the order of the three supplied load blocks and compare
  the native depth histories. Keep the material and geometry fixed.”
