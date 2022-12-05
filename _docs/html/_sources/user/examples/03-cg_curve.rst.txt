.. _3. Crack growth curve:

3. Crack growth curve
=====================

Crack growth curve equations have been extensively studied since the '50s of
the last century, almost in conjunction with the first catastrophic accident of
the first jet transportation aircraft, the De Havilland Comet 1 (1954), where
the propagation of fatigue cracks in the upper fuselage panels, starting from
the sharp corner of the top rectangular

windows brought 35 deaths. As Frost and Dugdale already pointed out in 1958,
the cylindrical specimens were the protagonists of the first half of the 20th
century in the panorama of fatigue, and this typology of specimens makes really
complex the study of crack growth, meaning that no special attention up to that
moment was given to crack growth testing of sheet specimens. Secondly, Frost
and Dugdale observed that “In aircraft structures the 'fail-safe' design
philosophy requires the structure to be constructed in such a way that fatigue
cracks do not cause catastrophic failure before corrective measures can be
taken”. The first scientist that emphasized the necessity to have polished 
surfaces to minimize the hotspots from where a crack can propagate is Griffith
who extended the theorem of minimum potential energy to the phenomena of 
rupture of elastic solids. Griffith's work was motivated by Inglis' linear
elastic solution for the stress around an elliptical hole asymptotically loaded
in tension, from which he predicted that the stress would go to infinite as the
ratio between the minor and major axis goes to zero. Griffith's theory provides
correct predictions as long as brittle materials, such as glass or ceramics,
are considered. Starting from the pioneering work of Griffith, Berto and
Lazzarin provided an exhaustive overview of local approaches for the
description of brittle and quasi-brittle fracture of engineering materials. 
Anyway, since in structural materials there is almost always some inelastic 
deformations around the crack faces, Griffith's hypothesis of linear elastic
medium in structural metals application becomes highly unrealistic. For this
reason, the first crack growth equation relating the stress with the crack 
growth rate (i.e. the crack length increment per cycle) did not make use of
the elastic energy approach. The formulation dates 1953 and has been proposed 
by Head; it is based on Inglis' solution and the final simplified form of the
crack growth equation is

.. math::

    \frac{da}{dN} = \phi \left( \Delta \sigma, \sigma_{Y} , \sigma'_{f} \right)
        \cdot a^{\frac{3}{2}} \cdot t_{p}^{\frac{3}{2}}

Where :math:`N` is the number of cycles, :math:`\phi(\sigma)` is
(asymptotically) a linear function of the stress, yielding, and strength,
:math:`a` is the half crack size, and :math:`t_p` is the thickness of the
plastic zone ahead of the crack tip. Frost and Dugdale argued that :math:`t_p`
is not a constant independent of crack length and derived the exponential model
for crack propagation, seldom used up to nowadays

.. math::

    \begin{align}
    \begin{cases}
    \frac{da}{dN} &= k_{FD} \left( (\Delta \sigma)^3 \right) \cdot a \\
    \ln\left(\frac{a}{a_i}\right) &= k_{FD} \left( (\Delta \sigma)^3 \right) \cdot N \\
    \end{cases}
    \end{align}

In which ai is the initial size of the crack and kFD is an experimental
quantity depending on the cubic power of the remote stress range
:math:`\Delta \sigma`. During WWII, a group of researchers of the U.S. Naval
Research Labs headed by George Rankine Irwin realized that plasticity plays an
important role in in fracture mechanics. On this purpose, Griffith's energy
formulation was modified in order to make it account for plasticity, too, i.e.
the energy release was redefined by adding a plastic dissipation term. Another
major achievement of Irwin's work is certainly the relation between the energy
release rate :math:`\hat{G}` and the stress intensity factor in opening mode
:math:`K_I`:

.. math::

    \hat{G} = K_I^2 / E^{\star}

where :math:`E^{\star}=E` for plain stress or :math:`E/(1-\nu^2)` for plane
strain. The critical stress intensity factor is the value of :math:`K` beyond
which a crack starts to propagate and is addressed as fracture toughness
:math:`K_C`. Namely, the toughness is the resistance to fracture of a material,
it is a material property and is defined as the stress intensity factor
required for a crack to advance from length a to a C. The fracture toughness
values have been grouped by material family by Ashby and are shown here below.

.. image:: ../../_static/_img/toughness-yielding.png

Few years later, in 1963, Paris and Erdogan published a work substantiated by
many experimental tests where they postulated, differently from Head or Frost
and Dugdale, that the crack growth is described by a power law of the stress
intensity factor, viz.

.. math::

    \frac{da}{dN} = C \left( \Delta K \right)^m

At the time of publication, the authors were uncertain on the value of the
exponent m, in fact there is a famous statement in their paper saying: *“The
authors are hesitant but cannot resist the temptation to draw the straight line
slope 1/4 through the data...”*. Therefore, the so-called Paris' law, or
Paris-Erdogan law, has been formulated in principle with fixed m=4. Indeed,
as evident from the figure here below (from Ashby's book), the majority of
engineering alloys is concentrated in the neighborhood of m=4, and this may
have tricked the authors.

.. image:: ../../_static/_img/paris-slope-ashby.png

Paris' law is considered valid within the range 
:math:`\Delta K_{th} < \Delta K < \Delta K_{cr}`, where :math:`\Delta K_{cr}`
is the critical stress intensity factor range which depends on the toughness
as :math:`\Delta K_{cr} = (1-R)K_{Ic}`, with R load ratio, and
:math:`\Delta K_{th}` is the threshold value below which the crack should not
propagate.

a. Definition of multiple Paris' laws 
-------------------------------------

.. note::
    In this example we define four SN curves for free corrosion as per 
    `DNVGL-RP-C203 <shorturl.at/ipBKL>`_ and plot them using
    `matplotlib <https://matplotlib.org>`_ and `plotly <https://plotly.com>`_. 
    We additionally define a random gaussian stress range-cycles history to 
    plot against the SN curves defined.

.. code-block:: python
    :linenos:

    # Paris' laws constants
    SIF = np.linspace(1,2500, 300)
    SLOPE_1 = np.array([2.88, 8.16])
    INTERCEPT_1 = np.array([1.21E-16, 3.98E-29])
    SLOPE_2 = np.array([8.16, 2.88])
    INTERCEPT_2 = np.array([1.21E-26, 3.98E-13])
    SLOPE_3 = 8.16
    INTERCEPT_3 = 1.21E-26
    THRESHOLD = 75.
    CRITICAL = 2000.

    # Paris' curves definition
    pc_1 = pf.ParisCurve(slope=SLOPE_1, intercept=INTERCEPT_1, norm="The norm",
                        environment="Environment", curve="nr. 1")
    pc_2 = pf.ParisCurve(slope=SLOPE_1, intercept=INTERCEPT_1,threshold=THRESHOLD,
                        critical=CRITICAL, norm="The norm",
                        environment="Environment", curve="nr. 2")
    pc_3 = pf.ParisCurve(slope=SLOPE_3, intercept=INTERCEPT_3, norm="The norm",
                     environment="Environment", curve="nr. 3")
    pc_4 = pf.ParisCurve(slope=SLOPE_2, intercept=INTERCEPT_2,threshold=THRESHOLD,
                        critical=CRITICAL, norm="The norm",
                        environment="Environment", curve="nr. 4")
    
    # Plotting
    fig, axs = plt.subplots(1,2, figsize=(12, 8))
    pc_1.plot(fig=fig, ax=axs[0],)
    pc_3.plot(fig=fig, ax=axs[0],)
    pc_2.plot(fig=fig, ax=axs[1],)
    pc_4.plot(fig=fig, ax=axs[1],)
    axs[0].legend()
    axs[1].legend()
    plt.show()

.. image:: ../../_static/_img/paris-curves.png

In the following years there have been many attempts to generalize Paris'
law, mainly to account for mean stress effect, crack closure and near
threshold/near failure modelling. The simplest model of Paris' law for mean 
stress effect has been proposed in 1970 by Walker:

.. math::

    \frac{da}{dN} = C_0 \left( \frac{\Delta K}{(1-R)^{1-\gamma}}\right)^m

With :math:`\gamma` being Walker exponent, :math:`R` the load ratio, and
:math:`C_0` being the intercept at :math:`R=0`.
