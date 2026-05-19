import numpy as np


def detect_pred_type(y_pred: np.ndarray, pred_type: str | None = None) -> str:
    """
    Detect whether y_pred contains discrete labels or probabilities.

    Parameters
    ----------
    y_pred : np.ndarray
        Predictions to inspect.
    pred_type : str | None
        If provided, skip detection and return this value directly.
        Must be 'labels' or 'proba'.

    Returns
    -------
    str
        'labels' or 'proba'.

    Raises
    ------
    ValueError
        If pred_type is invalid or y_pred cannot be identified.
    """
    if pred_type is not None:
        if pred_type not in {"labels", "proba"}:
            raise ValueError(
                f"pred_type must be 'labels' or 'proba', got '{pred_type}'."
            )
        return pred_type

    # Integer dtype → labels
    if np.issubdtype(y_pred.dtype, np.integer):
        return "labels"

    # Float dtype → check values
    if np.issubdtype(y_pred.dtype, np.floating):
        if y_pred.min() < 0.0 or y_pred.max() > 1.0:
            raise ValueError(
                "y_pred contains float values outside [0, 1]. "
                "Expected probabilities in [0, 1] or integer labels."
            )
        # Float values strictly in (0, 1) → proba
        # Float values only 0.0 and 1.0 → ambiguous, assume labels
        unique = np.unique(y_pred)
        if set(unique).issubset({0.0, 1.0}):
            return "labels"
        return "proba"

    raise ValueError(
        f"Cannot determine pred_type from y_pred with dtype {y_pred.dtype}. "
        "Pass pred_type='labels' or pred_type='proba' explicitly."
    )