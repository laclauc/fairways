Installation
============

Requirements
------------

- Python >= 3.10
- NumPy >= 2.0

Install from PyPI
-----------------

.. code-block:: bash

   pip install equiml

Optional dependencies
---------------------

For statistical tests (p-values, confidence intervals):

.. code-block:: bash

   pip install equiml[stats]

For visualizations:

.. code-block:: bash

   pip install equiml[viz]

Install all optional dependencies:

.. code-block:: bash

   pip install equiml[stats,viz]

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/laclauc/equiml
   cd equiml
   pip install -e ".[dev]"
