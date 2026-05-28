import numpy as np
from ..base import PreProcessor, PreProcessingResult
from ...metrics.discrete import _resolve_sensitive


def _compute_quantile_repair(
    X0: np.ndarray,
    X1: np.ndarray,
    lambda_: float,
    repair_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Repair two 1D arrays towards their quantile barycenter.

    Parameters
    ----------
    X0 : np.ndarray
        Feature values for group S=0.
    X1 : np.ndarray
        Feature values for group S=1.
    lambda_ : float
        Repair amount in [0, 1]. 0 = no repair, 1 = full repair.
    repair_type : str
        'total' or 'geometric'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Repaired arrays for group 0 and group 1.
    """
    n0, n1 = len(X0), len(X1)

    # Empirical quantile functions
    quantiles = np.linspace(0, 1, max(n0, n1) + 1)[1:]

    F0_inv = np.quantile(X0, quantiles)
    F1_inv = np.quantile(X1, quantiles)

    # Barycenter quantile function (median for 2 groups = mean)
    FB_inv = 0.5 * F0_inv + 0.5 * F1_inv

    def repair_group(X: np.ndarray, F_inv: np.ndarray) -> np.ndarray:
        """Map X towards the barycenter via quantile transport."""
        n = len(X)
        q = np.linspace(0, 1, n + 1)[1:]
        # Rank each value in its group
        ranks = np.argsort(np.argsort(X)) / n
        ranks = np.clip(ranks, 0, len(F_inv) - 1).astype(int)

        if repair_type == "total":
            # Full repair: map to barycenter
            return np.interp(
                np.argsort(np.argsort(X)) / (n - 1 + 1e-10),
                quantiles,
                FB_inv,
            )
        elif repair_type == "geometric":
            # Partial repair: linear interpolation between original and full repair
            fully_repaired = np.interp(
                np.argsort(np.argsort(X)) / (n - 1 + 1e-10),
                quantiles,
                FB_inv,
            )
            return (1 - lambda_) * X + lambda_ * fully_repaired
        else:
            raise ValueError(f"Unknown repair_type '{repair_type}'.")

    X0_repaired = repair_group(X0, F0_inv)
    X1_repaired = repair_group(X1, F1_inv)

    return X0_repaired, X1_repaired


class WassersteinRepair(PreProcessor):
    """
    Fair data repair using optimal transport theory.

    Transforms numerical features so that their conditional distributions
    given the sensitive attribute are mapped towards a common barycenter
    distribution, reducing disparate impact while preserving rank ordering.

    Two methods are available:

    - ``method='quantile'``: Operates feature-by-feature using empirical
      quantile functions. No additional dependencies required. Corresponds
      to the 1D repair of [FFM15]_ and the geometric repair of [GGL19]_.

    - ``method='ot'``: Uses optimal transport to repair all features jointly
      in the multidimensional case. Requires the ``POT`` package
      (``pip install equiml[ot]``). Corresponds to [GGL19]_.

    Three repair types are available via ``repair_type``:

    - ``'total'``: Full repair towards the Wasserstein barycenter.
      Guarantees statistical parity but may reduce accuracy (``lambda_=1``).
    - ``'geometric'``: Partial repair along the Wasserstein geodesic.
      Trade-off between fairness and accuracy controlled by ``lambda_``.
    - ``'random'``: Random repair as in [GGL19]_. Better control of
      total variation distance. Only available with ``method='ot'``.

    Parameters
    ----------
    repair_type : str, default='geometric'
        Type of repair: 'total', 'geometric', or 'random'.
    lambda_ : float, default=1.0
        Repair amount in [0, 1]. 0 = no repair, 1 = full repair.
        Only used for 'geometric' and 'random' repair types.
    method : str, default='quantile'
        Repair method: 'quantile' (numpy only) or 'ot' (requires POT).

    References
    ----------
    .. [FFM15] Feldman, M., Friedler, S.A., Moeller, J., Scheidegger, C.,
               & Venkatasubramanian, S. (2015). Certifying and removing
               disparate impact. In Proceedings of KDD (pp. 259-268).
               https://doi.org/10.1145/2783258.2783311

    .. [GGL19] Gordaliza, P., Del Barrio, E., Gamboa, F., & Loubes, J.M.
               (2019). Obtaining fairness using optimal transport theory.
               In Proceedings of ICML (pp. 2357-2365).
               https://arxiv.org/abs/1806.03195
    """

    def __init__(
        self,
        repair_type: str = "geometric",
        lambda_: float = 1.0,
        method: str = "quantile",
        weights: tuple[float, float] = (0.5, 0.5),
        random_state: int | None = None,
    ) -> None:
        if repair_type not in {"total", "geometric", "random"}:
            raise ValueError(
                f"repair_type must be 'total', 'geometric', or 'random', "
                f"got '{repair_type}'."
            )
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(
                f"lambda_ must be in [0, 1], got {lambda_}."
            )
        if method not in {"quantile", "ot"}:
            raise ValueError(
                f"method must be 'quantile' or 'ot', got '{method}'."
            )
        if method == "ot" and repair_type == "random":
            try:
                import ot  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Random repair with method='ot' requires the POT package. "
                    "Install it with: pip install equiml[ot]"
                )
        if method == "ot":
            try:
                import ot  # noqa: F401
            except ImportError:
                raise ImportError(
                    "method='ot' requires the POT package. "
                    "Install it with: pip install equiml[ot]"
                )

        self.repair_type = repair_type
        self.lambda_ = lambda_
        self.method = method
        self.weights = weights
        self.random_state = random_state
        self._groups: list = []
        self._fitted: bool = False

    @property
    def name(self) -> str:
        return "wasserstein_repair"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> "WassersteinRepair":
        """
        Fit the repair on training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            Must contain only numerical features.
        y : np.ndarray
            Ground truth labels. Not used — included for API consistency.
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        WassersteinRepair
            The fitted mitigator (self).
        """
        s = _resolve_sensitive(sensitive)
        self._groups = list(np.unique(s))
        if len(self._groups) != 2:
            raise ValueError(
                f"WassersteinRepair currently supports exactly 2 groups, "
                f"got {len(self._groups)}: {self._groups}."
            )
        self._sensitive_resolved = s
        self._X_fit = X
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
        Apply the repair to the feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Ground truth labels.
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        PreProcessingResult
            Repaired feature matrix with same y and no weights.
        """
        if not self._fitted:
            raise RuntimeError(
                "WassersteinRepair is not fitted yet. "
                "Call fit() before transform()."
            )

        s = _resolve_sensitive(sensitive)
        X_repaired = X.copy().astype(float)

        g0, g1 = self._groups
        mask0 = s == g0
        mask1 = s == g1

        if self.method == "quantile":
            # Repair each feature independently
            n_features = X.shape[1] if X.ndim > 1 else 1
            if X.ndim == 1:
                X_repaired = X_repaired.reshape(-1, 1)

            for j in range(n_features):
                X0_j = X_repaired[mask0, j]
                X1_j = X_repaired[mask1, j]

                repair_type = "total" if self.repair_type == "total" else "geometric"
                lambda_ = 1.0 if self.repair_type == "total" else self.lambda_

                X0_rep, X1_rep = _compute_quantile_repair(
                    X0_j, X1_j, lambda_, repair_type
                )
                X_repaired[mask0, j] = X0_rep
                X_repaired[mask1, j] = X1_rep

            if X.ndim == 1:
                X_repaired = X_repaired.ravel()

        elif self.method == "ot":
            from .wasserstein_repair_ot import _random_repair_ot
            X_input = X_repaired if X.ndim > 1 else X_repaired.reshape(-1, 1)
            X0_rep, X1_rep = _random_repair_ot(
                X_input[mask0],
                X_input[mask1],
                lambda_=self.lambda_,
                pi0=self.weights[0],
                pi1=self.weights[1],
                random_state=self.random_state,
            )
            X_repaired = X_input.copy().astype(float)
            X_repaired[mask0] = X0_rep
            X_repaired[mask1] = X1_rep
            if X.ndim == 1:
                X_repaired = X_repaired.ravel()

        return PreProcessingResult(
            X=X_repaired,
            y=y,
            extra={
                "repair_type": self.repair_type,
                "lambda_": self.lambda_,
                "method": self.method,
            },
        )