# equiml

[![CI](https://github.com/laclauc/equiml/actions/workflows/ci.yml/badge.svg)](https://github.com/laclauc/equiml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/equiml)](https://pypi.org/project/equiml/)
[![Python](https://img.shields.io/pypi/pyversions/equiml)](https://pypi.org/project/equiml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://laclauc.github.io/equiml/)

**Fairness auditing and mitigation for machine learning.**

## Installation

```bash
pip install equiml
```

## Quickstart

```python
import numpy as np
from equiml import audit

y_true    = np.array([1, 0, 1, 0, 1, 0])
y_pred    = np.array([1, 0, 0, 0, 1, 1])
sensitive = np.array([0, 0, 0, 1, 1, 1])

result = audit(y_true, y_pred, sensitive, metrics="all")
print(result)
```

## Features

- **Fairness metrics** — Demographic Parity, Equalized Odds, Equal Opportunity, Predictive Parity, AUC Parity, Calibration Parity, Brier Score Parity
- **Automatic detection** — labels or probability scores detected automatically
- **Intersectional analysis** — audit across combinations of sensitive attributes
- **Extensible** — add custom metrics by subclassing `FairnessMetric`
- **Pure NumPy** — compatible with scikit-learn, PyTorch, JAX, TensorFlow

## Documentation

Full documentation at [laclauc.github.io/equiml](https://laclauc.github.io/equiml/)

## License

MIT
