Quickstart
==========

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from equiml import audit

   result = audit(y_true, y_pred, sensitive, metrics="all")
   print(result)

Discrete predictions (labels)
------------------------------

.. code-block:: python

   import numpy as np
   from equiml import audit

   y_true    = np.array([1, 0, 1, 0, 1, 0])
   y_pred    = np.array([1, 0, 0, 0, 1, 1])
   sensitive = np.array([0, 0, 0, 1, 1, 1])

   result = audit(y_true, y_pred, sensitive, metrics="all")
   print(result)
   # AuditResult(pred_type='labels', groups=[0, 1])
   # Metrics:
   #   demographic_parity: 0.1667
   #   equal_opportunity: 0.5000
   #   equalized_odds: 0.5000
   #   predictive_parity: 0.5000

Probability scores
------------------

.. code-block:: python

   y_pred_proba = np.array([0.9, 0.2, 0.8, 0.4, 0.7, 0.3])

   result = audit(y_true, y_pred_proba, sensitive, metrics="all")
   # pred_type detected automatically as 'proba'

Selecting specific metrics
--------------------------

.. code-block:: python

   result = audit(y_true, y_pred, sensitive,
                  metrics=["demographic_parity", "equalized_odds"])

Intersectional analysis
-----------------------

.. code-block:: python

   gender = np.array([0, 0, 0, 1, 1, 1])
   age    = np.array([0, 1, 0, 1, 0, 1])

   result = audit(y_true, y_pred, sensitive=[gender, age])

Custom metrics
--------------

.. code-block:: python

   from equiml.metrics import FairnessMetric
   from equiml.metrics.base import MetricResult

   class DisparateImpact(FairnessMetric):

       @property
       def requires(self) -> set[str]:
           return {"labels"}

       @property
       def name(self) -> str:
           return "disparate_impact"

       def compute(self, y_true, y_pred, sensitive, **kwargs) -> MetricResult:
           groups = np.unique(sensitive)
           rates = {g: y_pred[sensitive == g].mean() for g in groups}
           values = list(rates.values())
           ratio = min(values) / max(values) if max(values) > 0 else 0.0
           return MetricResult(name=self.name, value=ratio, groups=rates)

   result = audit(y_true, y_pred, sensitive,
                  metrics=[DisparateImpact()])
