import numpy as np
from ..base import PreProcessor, PreProcessingResult
from ...metrics.discrete import _resolve_sensitive


def _random_repair_ot(
    X0: np.ndarray,
    X1: np.ndarray,
    lambda_: float,
    pi0: float,
    pi1: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random Repair via optimal transport (Gordaliza et al. 2019).

    For each point, independently draws b ~ Bernoulli(lambda_).
    If b=1, the point is replaced by its barycentric transport target.
    If b=0, the point is kept unchanged.

    Parameters
    ----------
    X0 : np.ndarray of shape (n0, d)
        Feature matrix for group S=0.
    X1 : np.ndarray of shape (n1, d)
        Feature matrix for group S=1.
    lambda_ : float
        Repair probability in [0, 1].
    pi0 : float
        Barycenter weight for group 0.
    pi1 : float
        Barycenter weight for group 1.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Repaired arrays for group 0 and group 1.
    """
    try:
        import ot
    except ImportError:
        raise ImportError(
            "Random Repair with method='ot' requires the POT package. "
            "Install it with: pip install equiml[ot]"
        )

    rng = np.random.default_rng(random_state)
    n0, n1 = len(X0), len(X1)

    # Cost matrix: squared Euclidean distances
    C = np.sum((X0[:, None, :] - X1[None, :, :]) ** 2, axis=-1)

    # Uniform marginals
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1

    # Solve optimal transport — returns transport plan gamma (n0 x n1)
    gamma = ot.emd(a, b, C)

    # For each point in X0, compute its barycentric target:
    # x̃0_i = sum_j (gamma[i,j] / a[i]) * (pi0*x0_i + pi1*x1_j)
    # Simplified: weighted average of transported positions
    X0_repaired = X0.copy().astype(float)
    X1_repaired = X1.copy().astype(float)

    # Repair group 0
    for i in range(n0):
        row = gamma[i]
        if row.sum() == 0:
            continue
        # Expected barycenter target for point i
        weights = row / row.sum()
        bary_target = pi0 * X0[i] + pi1 * (weights @ X1)
        b_i = rng.binomial(1, lambda_)
        if b_i == 1:
            X0_repaired[i] = bary_target

    # Repair group 1
    for j in range(n1):
        col = gamma[:, j]
        if col.sum() == 0:
            continue
        weights = col / col.sum()
        bary_target = pi0 * (weights @ X0) + pi1 * X1[j]
        b_j = rng.binomial(1, lambda_)
        if b_j == 1:
            X1_repaired[j] = bary_target

    return X0_repaired, X1_repaired


class WassersteinRepairOT(PreProcessor):
    """
    Fair data repair using optimal transport (multidimensional case).

    Implements the Random Repair procedure of [GGL19]_, which transports
    the conditional feature distributions towards their Wasserstein
    barycenter using optimal transport.

    Unlike the quantile-based repair (which operates feature by feature),
    this method handles all features jointly, preserving their joint
    structure.

    For each sample, independently draws ``b ~ Bernoulli(lambda_)``:

    - If ``b=1``: the sample is replaced by its barycentric transport target.
    - If ``b=0``: the sample is kept unchanged.

    This guarantees that the dataset size is preserved and provides
    better control of the total variation distance between groups than
    the geometric repair, as shown empirically in [GGL19]_.

    Parameters
    ----------
    lambda_ : float, default=0.5
        Repair probability in [0, 1]. 0 = no repair, 1 = full repair.
    weights : tuple[float, float], default=(0.5, 0.5)
        Barycenter weights (pi0, pi1) for group 0 and group 1.
        Must sum to 1.
    random_state : int or None, default=None
        Random seed for reproducibility.

    References
    ----------
    .. [GGL19] Gordaliza, P., Del Barrio, E., Gamboa, F., & Loubes, J.M.
               (2019). Obtaining fairness using optimal transport theory.
               In Proceedings of ICML (pp. 2357-2365).
               https://arxiv.org/abs/1806.03195
    """

    def __init__(
        self,
        lambda_: float = 0.5,
        weights: tuple[float, float] = (0.5, 0.5),
        random_state: int | None = None,
    ) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}.")
        if len(weights) != 2:
            raise ValueError(f"weights must have 2 elements, got {len(weights)}.")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {sum(weights)}.")

        self.lambda_ = lambda_
        self.weights = weights
        self.random_state = random_state
        self._groups: list = []
        self._fitted: bool = False

    @property
    def name(self) -> str:
        return "wasserstein_repair_ot"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> "WassersteinRepairOT":
        """
        Fit the repair on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix. Must contain only numerical features.
        y : np.ndarray
            Ground truth labels. Not used directly.
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        WassersteinRepairOT
            The fitted mitigator (self).
        """
        s = _resolve_sensitive(sensitive)
        self._groups = list(np.unique(s))
        if len(self._groups) != 2:
            raise ValueError(
                f"WassersteinRepairOT supports exactly 2 groups, "
                f"got {len(self._groups)}: {self._groups}."
            )
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
        Apply Random Repair to the feature matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray
            Ground truth labels.
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        PreProcessingResult
            Repaired feature matrix, same y, no weights.
            Dataset size is preserved.
        """
        if not self._fitted:
            raise RuntimeError(
                "WassersteinRepairOT is not fitted yet. "
                "Call fit() before transform()."
            )

        s = _resolve_sensitive(sensitive)
        g0, g1 = self._groups
        mask0 = s == g0
        mask1 = s == g1

        X_input = X if X.ndim > 1 else X.reshape(-1, 1)

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
                "lambda_": self.lambda_,
                "weights": self.weights,
                "method": "ot",
                "repair_type": "random",
            },
        )
