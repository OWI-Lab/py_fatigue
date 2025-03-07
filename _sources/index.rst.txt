.. You can adapt this file completely to your liking, but it should at least
.. contain the root `toctree` directive.

.. Custom roles and commands

.. |_| unicode:: U+00A0 .. non-breaking space

.. role:: blue

py-fatigue documentation
========================

.. topic:: Release STABLE

   Version |release|

`py-fatigue` bundles the main functionality for performing cyclic stress
(fatigue and crack growth) analysis and cycle-counting. The toolbox is based 
on four buinding blocks:

1. `cycle_count`: collects all the functions related with cycle-counting fatigue data
2. `material`: allows the definition of some fatigue-specific material properties such as :term:`SN curve<SN Curve>` and :term:`Paris' law`.
3. `geometry`: enables the instantiation of specific crack geometries for crack propagation analysis.
4. `damage`: the module collecting the fatigue life calculation models:

   - `stress-life`: fatigue analysis based on the :term:`cycle-counting<Cycle-counting>` and :term:`SN curve<SN Curve>`.
   - `crack_growth`: crack propagation (growth) analysis based on  the :term:`cycle-counting<Cycle-counting>`, the :term:`Paris' law` and the geometrical definition of a crack case (defect geometry and medium).



.. grid:: 1 2 2 2
   :gutter: 3 3 4 5

   .. grid-item-card::

     .. raw:: html

        <html><i class="fa-solid fa-gamepad fa-lg"></i>&nbsp;&nbsp;&nbsp;&nbsp<b>Getting Started</b></html>

     New to `py-fatigue`? Stert here! The Absolute Beginner's Guide contains
     all the introductiory material to familiarize with `py-fatigue` main
     concepts and start playing with the examples.

     +++

     .. button-ref:: user/01-absolute-noob
        :expand:
        :color: primary
        :click-parent:

        To the absolute beginner's guide

   .. grid-item-card::

      .. raw:: html

         <html><i class="fa-solid fa-rocket fa-lg fa-beat"></i>&nbsp;&nbsp;&nbsp;&nbsp<b>Interactive Examples</b></html>

      The Binder Environment allows you to run the examples directly in your
      browser without installing anything on your local machine. The examples
      are based on Jupyter notebooks and can be modified and extended as needed.

      +++

      .. button-link:: https://mybinder.org/v2/gh/OWI-Lab/py-fatigue-tutorials/HEAD
         :expand:
         :color: primary
         :click-parent:

         To the Binder environment

   .. grid-item-card::

     .. raw:: html

        <html><i class="fa-solid fa-book fa-lg"></i>&nbsp;&nbsp;&nbsp;&nbsp<b>User Guide</b></html>

     The user guide provides in-depth information on the key concepts of 
     `py-fatigue` with useful background information and explanation.

     +++

     .. button-ref:: user/02-user-guide
        :expand:
        :color: primary
        :click-parent:

        To the user guide

   .. grid-item-card::

     .. raw:: html

        <html><i class="fa-solid fa-code fa-lg"></i>&nbsp;&nbsp;&nbsp;&nbsp<b>API Reference</b></html>

     The API reference provides detailed information on the methods
     and classes available in `py-fatigue`. The reference describes
     how tools work and which parameters can be used. It assumes that
     you have an understanding of the key concepts.

     +++

     .. button-ref:: api/01-index
        :expand:
        :color: primary
        :click-parent:

        To the reference guide

.. toctree::
   :caption: Beginner's guide
   :hidden:

   user/01-absolute-noob

.. toctree::
   :caption: User guide
   :hidden:

   user/02-user-guide

.. toctree::
   :caption: Glossary and guidelines
   :hidden:

   user/additional/01-index

.. toctree::
   :caption: API Reference
   :hidden:

   api/01-index

.. toctree::
   :caption: Change log
   :hidden:

   changelog

   

Miscellaneous Pages
+++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
