# API index

Start with the inputs available for your calculation. These links open the
class and function reference, including parameters and return values.

| Task | Start with | Inputs and result |
| --- | --- | --- |
| Count cycles from a measured signal | [`CycleCount`][py_fatigue.cycle_count.cycle_count.CycleCount] | Stress samples and optional time values produce ranges, means, counts, and residuals. |
| Estimate cumulative fatigue damage | [`SNCurve`][py_fatigue.material.sn_curve.SNCurve] and [`calc_pm`][py_fatigue.damage.stress_life.calc_pm] | Combine a material or weld detail curve with counted cycles to get a damage contribution per range. |
| Grow a crack from cycle-counted loading | [`ParisCurve`][py_fatigue.material.crack_growth_curve.ParisCurve], [`WalkerCurve`][py_fatigue.material.crack_growth_curve.WalkerCurve], and [`get_crack_growth`][py_fatigue.damage.crack_growth.get_crack_growth] | Combine cycle counts, a growth law, and a geometry to obtain crack propagation history. |
| Grow a crack under ordered loads | [`NasgroCurve`][py_fatigue.crack_growth.NasgroCurve], [`LoadBlock`][py_fatigue.crack_growth.LoadBlock], and [`simulate_crack_growth`][py_fatigue.crack_growth.simulate_crack_growth] | Supply stress extrema in order and obtain accepted crack profiles, per-tip stress intensities, and the stop reason. |

## Modules

- [Cycle counts](cycle_count/cycle_count_.md), [histograms](cycle_count/histogram.md), and [rainflow](cycle_count/rainflow.md)
- [S–N curves](material/sn_curve.md) and [crack-growth curves](material/crack_growth_curve.md)
- [Generic crack geometries](geometry/generic.md) and [cylinders](geometry/cylinder.md)
- [Stress ranges](stress/stress_range.md) and [mean stress corrections](stress/mean_stress.md)
- [Stress–life damage](damage/stress_life.md), [cycle-counted crack growth](damage/crack_growth.md), and [ordered crack growth](damage/ordered_crack_growth.md)
- [Synthetic signals](testing.md), [utilities](utils.md), and [plot styling](styling.md)

For a complete calculation, use the [beginner’s guide](../user/01-absolute-noob.md)
or an [interactive tutorial](../user/examples/01-notebook_setup.md).
