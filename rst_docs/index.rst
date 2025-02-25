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
      :octicon:`rocket` |_| |_| |_| |_| New to `py-fatigue`? Start here!

      New to `py-fatigue`? Check out the Absolute Beginner's Guide. It contains
      an introduction to `py-fatigue`'s main concepts and links to additional
      tutorials.

      +++

      .. button-ref:: user/01-absolute-noob
         :expand:
         :color: primary
         :click-parent:

         To the absolute beginner's guide

   .. grid-item-card::
      :octicon:`repo` |_| |_| |_| |_| User's Guide

      The user guide provides in-depth information on the key concepts of 
      `py-fatigue` with useful background information and explanation.

      +++

      .. button-ref:: user/02-user-guide
         :expand:
         :color: primary
         :click-parent:

         To the user guide

   .. grid-item-card::
      :octicon:`code` |_| |_| |_| |_| API Reference

      The reference guide contains a detailed description of the functions,
      modules, and objects included in `py-fatigue`. The reference describes
      how the methods work and which parameters can be used. It assumes that
      you have an understanding of the key concepts.

      +++

      .. button-ref:: api/01-index
         :expand:
         :color: primary
         :click-parent:

         To the reference guide

   .. grid-item-card::
      :octicon:`link` |_| |_| |_| |_| Interactive Examples

      The Binder Environment allows you to run the examples directly in your
      browser without installing anything on your local machine. The examples
      are based on Jupyter notebooks and can be modified and extended as needed.

      +++

      .. button-link:: https://mybinder.org/v2/gh/OWI-Lab/py-fatigue-tutorials/HEAD
         :expand:
         :color: primary
         :click-parent:

         To the Binder environment


.. toctree::
   :caption: Beginner's guide
   :hidden:

   user/01-absolute-noob

.. toctree::
   :caption: User's guide
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
