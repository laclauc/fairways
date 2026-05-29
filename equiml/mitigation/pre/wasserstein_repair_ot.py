import numpy as np


def _random_repair_ot(
    X0: np.ndarray,
    X1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    s0_val,
    s1_val,
    lambda_: float,
    pi0: float,
    pi1: float,
    random_state: int | None = None,
) -> tuple:
    """
    Random Repair via optimal transport (Gordaliza et al. 2019).

    Implements Algorithm from Section 5.1.2 of [GGL19]_.

    For each point x_{s,i}:
    - Draw b_i ~ Bernoulli(lambda_)
    - If b_i = 0: keep x_{s,i} unchanged (weight = 1.0)
    - If b_i = 1: split into {pi0*x0_i + pi1*x1_j} for all j with gamma_ij > 0
                  each with weight gamma_ij * n0 (renormalized)

    Parameters
    ----------
    X0 : np.ndarray of shape (n0, d)
        Feature matrix for group S=s0_val.
    X1 : np.ndarray of shape (n1, d)
        Feature matrix for group S=s1_val.
    y0 : np.ndarray of shape (n0,)
        Labels for group S=s0_val.
    y1 : np.ndarray of shape (n1,)
        Labels for group S=s1_val.
    s0_val : scalar
        Sensitive attribute value for group 0.
    s1_val : scalar
        Sensitive attribute value for group 1.
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
    tuple of 9 arrays:
        X0_rep, y0_rep, w0_rep, s0_rep,
        X1_rep, y1_rep, w1_rep, s1_rep,
        info (dict with n0_repaired, n1_repaired)

    References
    ----------
    .. [GGL19] Gordaliza, P., Del Barrio, E., Gamboa, F., & Loubes, J.M.
               (2019). Obtaining fairness using optimal transport theory.
               In Proceedings of ICML (pp. 2357-2365).
               https://arxiv.org/abs/1806.03195
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

    # Solve optimal transport — transport plan gamma (n0 x n1)
    gamma = ot.emd(a, b, C)

    # --- Repair group 0 ---
    X0_new, y0_new, w0_new, s0_new = [], [], [], []
    bernoulli_0 = rng.binomial(1, lambda_, size=n0)

    for i in range(n0):
        if bernoulli_0[i] == 0:
            X0_new.append(X0[i])
            y0_new.append(y0[i])
            w0_new.append(1.0)
            s0_new.append(s0_val)
        else:
            for j in range(n1):
                if gamma[i, j] > 0:
                    x_tilde = pi0 * X0[i] + pi1 * X1[j]
                    X0_new.append(x_tilde)
                    y0_new.append(y0[i])
                    w0_new.append(gamma[i, j] * n0)
                    s0_new.append(s0_val)

    # --- Repair group 1 ---
    X1_new, y1_new, w1_new, s1_new = [], [], [], []
    bernoulli_1 = rng.binomial(1, lambda_, size=n1)

    for j in range(n1):
        if bernoulli_1[j] == 0:
            X1_new.append(X1[j])
            y1_new.append(y1[j])
            w1_new.append(1.0)
            s1_new.append(s1_val)
        else:
            for i in range(n0):
                if gamma[i, j] > 0:
                    x_tilde = pi0 * X0[i] + pi1 * X1[j]
                    X1_new.append(x_tilde)
                    y1_new.append(y1[j])
                    w1_new.append(gamma[i, j] * n1)
                    s1_new.append(s1_val)

    info = {
        "n0_repaired": len(X0_new),
        "n1_repaired": len(X1_new),
    }

    return (
        np.array(X0_new),
        np.array(y0_new),
        np.array(w0_new),
        np.array(s0_new),
        np.array(X1_new),
        np.array(y1_new),
        np.array(w1_new),
        np.array(s1_new),
        info,
    )