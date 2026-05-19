from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetricResult:
    """
    Result of a fairness metric computation.

    Attributes
    ----------
    name : str
        Name of the metric.
    value : float
        Main scalar value of the metric (e.g. gap between groups).
    groups : dict[str, float]
        Per-group breakdown of the metric value.
    confidence_interval : tuple[float, float] | None
        (lower, upper) confidence interval for the metric value, if computed.
    p_value : float | None
        p-value from statistical test, if computed.
    curve : np.ndarray | None
        Curve data (e.g. ROC curve, calibration curve), if applicable.
    extra : dict[str, Any]
        Any additional metric-specific information.
    """
    name: str
    value: float
    groups: dict[str, float] = field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    curve: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        ci = (
            f", CI=({self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f})"
            if self.confidence_interval is not None
            else ""
        )
        pval = f", p={self.p_value:.3f}" if self.p_value is not None else ""
        return f"MetricResult({self.name}={self.value:.4f}{ci}{pval})"


class FairnessMetric(ABC):
    """Base class for all fairness metrics."""

    @abstractmethod
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """
        Compute the fairness metric.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Predictions (discrete labels or probabilities).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s). Pass a single np.ndarray for a single
            attribute, or a list of np.ndarray for intersectional analysis.
        **kwargs
            Additional metric-specific parameters.

        Returns
        -------
        MetricResult
            Rich result object containing value, groups, CI, p-value, curve.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def requires(self) -> set[str]:
        """
        Return the set of prediction types required by this metric.

        Returns
        -------
        set[str]
            Subset of {'labels', 'proba'}.

        Examples
        --------
        >>> metric.requires
        {'labels'}
        >>> metric.requires
        {'labels', 'proba'}
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Metric name — defaults to lowercase class name."""
        return self.__class__.__name__.lower()