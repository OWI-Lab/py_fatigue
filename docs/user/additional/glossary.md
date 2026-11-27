# Glossary

------------------------------------------------------------------------

## Fatigue

Progressive deterioration of a material under cyclic loading. See
[fatigue loading in wind turbines](02-rainflow-and-residuals.md#fatigue-loading-in-wind-turbines).

## Stress ranges

The stress range is identified as Δσ and it is the difference between the maximum and the minimum stress in the fatigue cycle

$$\Delta\sigma = \sigma_{max} - \sigma_{min}$$

## Stress Amplitudes

Half of the stress range

$$\sigma_{alt} = \frac{\sigma_{max} - \sigma_{min}}{2}$$

## Mean stress

The mean stress is the average of the maximum and minimum stresses in the fatigue cycle.

$$\sigma_{mean} = \frac{(\sigma_{max} + \sigma_{min})}{2}$$

## Load ratio

The load ratio is the minimum stress divided by the maximum stress. At a
fixed crack size under proportional mode-I loading, the corresponding ratio
is $K_\min/K_\max$.

$$R = \frac{\sigma_{\min}}{\sigma_{\max}}$$

## SIF range

The mode-I stress intensity factor $K$ describes the near-tip opening
stress field. It has units of stress times the square root of length, such
as MPa √mm. Its range is $\Delta K = K_\max - K_\min$.

$$\Delta K = Y(a)\,\Delta\sigma\sqrt{\pi a}$$

Here $Y(a)$ is the dimensionless geometry factor, $a$ is the crack dimension
defined by the selected solution, and $\Delta\sigma$ is the applied stress
range. A surface-crack solution can give a different $K$ at each front point.

## Crack growth rate

The crack growth rate $da/dN$ is the advance of a crack dimension per cycle,
for example in mm/cycle. Py-Fatigue v3 provides Paris, Walker and NASGRO
growth laws. Use material coefficients calibrated in units consistent with
the analysis.

## Paris' law

$$\frac{da}{dN} = C(\Delta K)^m$$

The exponent $m$ is the slope in log–log coordinates. $C$ is the coefficient
in the chosen unit system. A Walker model adjusts the growth rate for the
load ratio. See the [crack-growth curve guide](../examples/02-crack-growth-curves.md).

## NASGRO growth law

A crack-growth law that uses the crack size and the minimum and maximum
mode-I stress intensities. See [how the NASGRO equation works](03-understand-ordered-crack-growth.md#nasgro-equation).

<span id="nasgro-equation"></span>
<span id="threshold-crack-opening-and-toughness"></span>
<span id="from-geometry-and-load-to-stress-intensity"></span>
<span id="advancing-two-front-dimensions"></span>

## Crack geometry and front

The geometry defines the flaw shape. The front is the boundary of the
crack within the material. See [crack geometry and intensity](03-understand-ordered-crack-growth.md#crack-geometry-and-front).

## Ordered load block and termination

A load block is a repeated pair of stress extrema and its cycle count.
A termination records why the ordered solver stopped. See
[ordered loads and stop reasons](03-understand-ordered-crack-growth.md#ordered-load-block-and-termination).

## Tabulated geometry factor

[`TabulatedGeometryFactor`][py_fatigue.geometry.generic.TabulatedGeometryFactor]
interpolates a dimensionless $Y$ against one growing crack dimension in
[`get_crack_growth`][py_fatigue.damage.crack_growth.get_crack_growth]. The
last tabulated depth is a geometry limit. See the
[MERC-3 comparison](../../tutorials/40-afgrow_merc3_longeron.html).

## Cycle-counting

Extracting fatigue cycles and their stress ranges from a stress history.

## Rainflow

A family of cycle-counting methods that use ordered stress reversals.
See the [rainflow explanation and worked trace](02-rainflow-and-residuals.md#rainflowchart).

<span id="rainflowchart"></span>

## Damage Equivalent Stress Range

The damage equivalent stress range (DES) represents linear fatigue damage
as a constant stress range. Applying $N_{eq}$ cycles at the DES produces
the same damage as the original signal.

We use the definition by [ECN](https://publicaties.ecn.nl/PdfFetch.aspx?nr=ECN-C--95-074).

$$DES = \left(\frac{\sum_in_i\cdot{}\Delta{}\sigma{}_i^m}{N_{eq}}\right)^{1/m}$$

## SN Curve

An S–N or Wöhler curve gives the fatigue life in cycles at a given
stress range. On one branch of an `SNCurve`, its base-10 intercept $b$
and slope $m$ give

$$
\log_{10} N = b - m\log_{10}\Delta\sigma,
\qquad N = 10^b (\Delta\sigma)^{-m}.
$$

For a multi-branch curve, use the slope and intercept of the branch
containing the stress range. See the [S–N curve guide](../examples/01-sn-curves.md).

## Damage Equivalent Moment

The damage equivalent moment applies the same principle as the
[damage equivalent stress range](#damage-equivalent-stress-range) to bending
moment.

## Palmgren-Miner Rule

Rule for linear fatigue, uses the idea of [SN curve](glossary.md) to determine the fatigue damage of a material or detail under the loading.

$$D = \sum_i(\frac{n_i}{N_i(\Delta{}\sigma_i)})$$

with $n_i$ the number of cycles of a given stress range $\sigma_i$. $N_i$ is the number of permisable cycles according to the [SN Curve](glossary.md).

## Residuals

The ordered turning points left when rainflow counting has closed all
cycles it can in the record. Adjacent residual points define half-cycles.

## Half-cycles

An open stress excursion counted with a weight of 0.5. See [residuals](#residuals).

## SHM

[Structural health monitoring](https://en.wikipedia.org/wiki/Structural_health_monitoring).

## Timestamp

A timestamp is a time information that is associated with a particular segmented data series. Usually it is represented as a [datetime.datetime](https://docs.python.org/3/library/datetime.html) instance.

## LFFD

Low-frequency fatigue dynamics. The term refers to larger, slower
load excursions that can span multiple recording windows. See
[how LFFD recovery combines residuals](02-rainflow-and-residuals.md#lffd).
