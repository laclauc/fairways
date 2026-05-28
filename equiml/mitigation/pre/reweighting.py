import numpy as np
from ..base import PreProcessor, PreProcessingResult
from ...metrics.discrete import _resolve_sensitive


class Reweighting(PreProcessor):
    """
    Reweighting for fairness-aware classification.

    Assigns sample weights to training examples so that each
    (group, label) combination is represented according to the
    expected distribution under independence between the sensitive
    attribute and the label.

    The weight for a sample with sensitive attribute S=s and label Y=y is:

    .. math::

        W(s, y) = \\frac{P(S=s) \\cdot P(Y=y)}{P(S=s, Y=y)}

    Under a fair model, S and Y would be independent, so the joint
    probability would equal the product of marginals. The weights
    correct for the observed dependence.

    References
    ----------
    .. [RW1] Kamiran, F., & Calders, T. (2012). Data preprocessing
             techniques for classification without discrimination.
             Knowledge and Information Systems, 33(1), 1-33.
             https://doi.org/10.1007/s10115-011-0463-8
    """

    def __init__(self) -> None:
        self._weights_map: dict = {}
        self._fitted: bool = False

    @property
    def name(self) -> str:
        return "reweighting"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> "Reweighting":
        """
        Compute sample weights from training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            Not used directly — included for API consistency.
        y : np.ndarray
            Ground truth binary labels of shape (n_samples,).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        Reweighting
            The fitted mitigator (self).
        """
        s = _resolve_sensitive(sensitive)
        n = len(y)

        groups = np.unique(s)
        labels = np.unique(y)

        # Marginal probabilities
        p_s = {g: np.mean(s == g) for g in groups}
        p_y = {lbl: np.mean(y == lbl) for lbl in labels}

        # Joint probabilities
        p_sy = {}
        for g in groups:
            for lbl in labels:
                p_sy[(g, lbl)] = np.mean((s == g) & (y == lbl))

        # Weights: W(s, y) = P(S) * P(Y) / P(S, Y)
        self._weights_map = {}
        for g in groups:
            for lbl in labels:
                joint = p_sy[(g, lbl)]
                if joint > 0:
                    self._weights_map[(g, lbl)] = (p_s[g] * p_y[lbl]) / joint
                else:
                    self._weights_map[(g, lbl)] = 1.0

        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> PreProcessingResult:
        """
        Apply sample weights to training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Ground truth labels of shape (n_samples,).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        PreProcessingResult
            Same X and y with computed sample weights.
        """
        if not self._fitted:
            raise RuntimeError(
                "Reweighting is not fitted yet. Call fit() before transform()."
            )

        s = _resolve_sensitive(sensitive)
        weights = np.array([
            self._weights_map.get((s[i], y[i]), 1.0)
            for i in range(len(y))
        ])

        return PreProcessingResult(
            X=X,
            y=y,
            weights=weights,
            extra={"weights_map": self._weights_map},
        )
