from .audit import audit
from .result import AuditResult
from .metrics.base import FairnessMetric, MetricResult

__version__ = "0.1.0"

__all__ = [
    "audit",
    "AuditResult",
    "FairnessMetric",
    "MetricResult",
]
