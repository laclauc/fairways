import numpy as np

from .detection import detect_pred_type
from .metrics.base import FairnessMetric, MetricResult
from .metrics.registry import METRICS_REGISTRY, LABELS_METRICS, PROBA_METRICS
from .result import AuditResult


def _resolve_metrics(
    metrics: str | list,
    pred_type: str,
) -> list[FairnessMetric]:
    """
    Resolve the metrics argument into a list of FairnessMetric instances.

    Parameters
    ----------
    metrics : str or list
        'all', a list of metric name strings, or a list of FairnessMetric
        instances (or a mix of both).
    pred_type : str
        'labels' or 'proba' — used to filter compatible metrics when
        metrics='all'.

    Returns
    -------
    list[FairnessMetric]
        Ready-to-use metric instances.

    Raises
    ------
    ValueError
        If a string metric name is not found in the registry, or if a
        metric requires a pred_type incompatible with the detected one.
    TypeError
        If an element of metrics is neither a string nor a FairnessMetric.
    """
    if metrics == "all":
        pool = LABELS_METRICS if pred_type == "labels" else PROBA_METRICS
        return [cls() for cls in pool.values()]

    resolved = []
    for m in metrics:
        if isinstance(m, str):
            if m not in METRICS_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{m}'. "
                    f"Available metrics: {list(METRICS_REGISTRY.keys())}."
                )
            resolved.append(METRICS_REGISTRY[m]())
        elif isinstance(m, FairnessMetric):
            resolved.append(m)
        else:
            raise TypeError(
                f"metrics must contain strings or FairnessMetric instances, "
                f"got {type(m)}."
            )

    # Validate compatibility with pred_type
    for m in resolved:
        if pred_type == "labels" and "labels" not in m.requires:
            raise ValueError(
                f"Metric '{m.name}' requires probabilities but y_pred "
                f"was detected as labels. Pass pred_type='proba' explicitly "
                f"or provide probability scores."
            )
        if pred_type == "proba" and "proba" not in m.requires:
            raise ValueError(
                f"Metric '{m.name}' requires labels but y_pred "
                f"was detected as probabilities. Pass pred_type='labels' "
                f"explicitly or provide discrete predictions."
            )

    return resolved


def audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray | list[np.ndarray],
    metrics: str | list = "all",
    pred_type: str | None = None,
) -> AuditResult:
    """
    Audit a model for fairness across sensitive groups.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Model predictions — discrete labels {0, 1} or probabilities [0, 1].
        Type is detected automatically unless pred_type is specified.
    sensitive : np.ndarray or list[np.ndarray]
        Sensitive attribute(s). Pass a single np.ndarray for a single
        attribute, or a list of np.ndarray for intersectional analysis.
    metrics : str or list, default='all'
        Metrics to compute. Options:
        - 'all': compute all metrics compatible with the detected pred_type.
        - list of strings: e.g. ['demographic_parity', 'auc_parity'].
        - list of FairnessMetric instances: for custom metrics.
        - mixed list of strings and FairnessMetric instances.
    pred_type : str or None, default=None
        Override automatic detection. Must be 'labels' or 'proba'.

    Returns
    -------
    AuditResult
        Object containing computed metrics, detected pred_type, and groups.

    Examples
    --------
    >>> import numpy as np
    >>> from equiml import audit
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 0, 0, 1, 1])
    >>> sensitive = np.array([0, 0, 0, 1, 1, 1])
    >>> result = audit(y_true, y_pred, sensitive)
    >>> result.metrics["demographic_parity"].value
    """
    # --- Input validation ---
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    if isinstance(sensitive, list) and len(sensitive) > 0 and isinstance(sensitive[0], np.ndarray):
        for i, s in enumerate(sensitive):
            s = np.asarray(s)
            if s.shape[0] != y_true.shape[0]:
                raise ValueError(
                    f"sensitive[{i}] must have the same length as y_true, "
                    f"got {s.shape[0]} and {y_true.shape[0]}."
                )
    else:
        sensitive = np.asarray(sensitive)
        if sensitive.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"sensitive must have the same length as y_true, "
                f"got {sensitive.shape[0]} and {y_true.shape[0]}."
            )

    # --- Detection ---
    detected_pred_type = detect_pred_type(y_pred, pred_type)

    # --- Resolve metrics ---
    metric_instances = _resolve_metrics(metrics, detected_pred_type)

    # --- Compute ---
    results: dict[str, MetricResult] = {}
    for metric in metric_instances:
        results[metric.name] = metric.compute(y_true, y_pred, sensitive)

    # --- Groups ---
    if isinstance(sensitive, list):
        from .metrics.discrete import _resolve_sensitive
        groups = list(np.unique(_resolve_sensitive(sensitive)))
    else:
        groups = list(np.unique(sensitive))

    return AuditResult(
        metrics=results,
        pred_type=detected_pred_type,
        groups=groups,
    )