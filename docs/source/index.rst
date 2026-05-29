equiml — Fairness Auditing and Mitigation for Machine Learning
==============================================================

**equiml** is a Python library for auditing and mitigating bias in machine learning models.

.. code-block:: python

   import numpy as np
   from equiml import audit

   result = audit(y_true, y_pred, sensitive, metrics="all")
   print(result)

Key features:

- **Fairness metrics** — Demographic Parity, Equalized Odds, AUC Parity, Calibration Parity and more
- **Automatic detection** — labels or probability scores detected automatically
- **Intersectional analysis** — audit across combinations of sensitive attributes
- **Extensible** — add custom metrics by subclassing ``FairnessMetric``
- **Pure NumPy** — compatible with scikit-learn, PyTorch, JAX, TensorFlow

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   concepts

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/audit
   api/metrics
   api/result
   api/mitigation

.. toctree::
   :maxdepth: 2
   :caption: Examples

   notebooks/01_quickstart

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`