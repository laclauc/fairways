Concepts
========

What is fairness in machine learning?
--------------------------------------

A model is considered **biased** when its predictions systematically
disadvantage certain groups defined by sensitive attributes such as
gender, age, or ethnicity.

equiml provides tools to **measure** and **mitigate** this bias.

Sensitive attributes
--------------------

A sensitive attribute is a feature that defines groups across which
fairness should be evaluated. Examples include gender, age group,
or ethnicity.

equiml accepts sensitive attributes as NumPy arrays:

.. code-block:: python

   sensitive = np.array([0, 0, 1, 1, 0, 1])  # 0 = group A, 1 = group B

For intersectional analysis, pass a list of arrays:

.. code-block:: python

   sensitive = [gender_array, age_array]

Fairness metrics
----------------

equiml implements two families of metrics.

**Metrics on discrete predictions (labels)**

Demographic Parity
~~~~~~~~~~~~~~~~~~

Measures the gap in positive prediction rates between groups.

.. math::

   \Delta_{DP} = P(\hat{Y}=1 | S=a) - P(\hat{Y}=1 | S=b)

A value of 0 indicates perfect demographic parity.

Equal Opportunity
~~~~~~~~~~~~~~~~~

Measures the gap in True Positive Rates (TPR) between groups.

.. math::

   \Delta_{EO} = TPR(S=a) - TPR(S=b)

Equalized Odds
~~~~~~~~~~~~~~

Measures both TPR and FPR gaps between groups.

.. math::

   \Delta_{EOdds} = \max(\Delta_{TPR}, \Delta_{FPR})

Predictive Parity
~~~~~~~~~~~~~~~~~

Measures the gap in precision between groups.

.. math::

   \Delta_{PP} = Precision(S=a) - Precision(S=b)

**Metrics on probability scores**

AUC Parity
~~~~~~~~~~

Measures the gap in ROC AUC scores between groups.

Calibration Parity
~~~~~~~~~~~~~~~~~~

Measures the gap in Expected Calibration Error (ECE) between groups.

Brier Score Parity
~~~~~~~~~~~~~~~~~~

Measures the gap in mean squared error between predicted probabilities
and true labels across groups.

The impossibility theorem
-------------------------

It is mathematically impossible to satisfy all fairness metrics
simultaneously when base rates differ across groups (Chouldechova 2017,
Kleinberg et al. 2016).

This means that **the choice of metric matters** — different metrics
can lead to different conclusions about whether a model is fair.

equiml lets you compute all metrics at once so you can make an
informed decision.
