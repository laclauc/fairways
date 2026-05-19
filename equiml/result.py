from dataclasses import dataclass
from .metrics.base import MetricResult


@dataclass
class AuditResult:
    """
    Result of a fairness audit.

    Attributes
    ----------
    metrics : dict[str, MetricResult]
        Computed metrics, keyed by metric name.
    pred_type : str
        Detected or specified prediction type: 'labels' or 'proba'.
    groups : list
        Unique groups found in the sensitive attribute(s).

    Examples
    --------
    >>> result = audit(y_true, y_pred, sensitive)
    >>> result.metrics["demographic_parity"].value
    0.12
    >>> result.groups
    [0, 1]
    >>> result.pred_type
    'labels'
    """

    metrics: dict[str, MetricResult]
    pred_type: str
    groups: list

    def __repr__(self) -> str:
        lines = [
            f"AuditResult(pred_type='{self.pred_type}', groups={self.groups})",
            "Metrics:",
        ]
        for name, result in self.metrics.items():
            lines.append(f"  {name}: {result.value:.4f}")
        return "\n".join(lines)
