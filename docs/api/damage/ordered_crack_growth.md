# Ordered crack growth

!!! abstract "Overview"

    Use `simulate_crack_growth` with a `NasgroCurve`, a crack geometry, and
    an ordered sequence of tensile `LoadBlock` instances. The result records
    accepted crack dimensions, stress intensities, and the reason the run stopped.
    See the [geometry and loading guide](../../user/examples/07-ordered-crack-growth.md)
    for units, supported cases, and solver controls.

::: py_fatigue.crack_growth
