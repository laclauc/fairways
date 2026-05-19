import numpy as np
from .base import FairnessMetric, MetricResult
from .discrete import _resolve_sensitive


def _check_proba(y: np.ndarray, name: str) -> None:
    """Raise if array values are not in [0, 1]."""
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(
            f"{name} must contain probabilities in [0, 1], "
            f"got min={y.min():.4f}, max={y.max():.4f}."
        )


def _roc_curve_per_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitive: np.ndarray,
) -> dict:
    """
    Compute ROC curve and AUC per group.

    Returns
    -------
    dict
        Keys are group values, values are dicts with
        'fpr', 'tpr', 'auc'.
    """
    groups = np.unique(sensitive)
    results = {}
    for g in groups:
        mask = sensitive == g
        yt = y_true[mask]
        ys = y_score[mask]

        thresholds = np.sort(np.unique(ys))[::-1]
        fprs, tprs = [0.0], [0.0]

        n_pos = np.sum(yt == 1)
        n_neg = np.sum(yt == 0)

        for t in thresholds:
            y_pred = (ys >= t).astype(int)
            tp = np.sum((yt == 1) & (y_pred == 1))
            fp = np.sum((yt == 0) & (y_pred == 1))
            tprs.append(tp / n_pos if n_pos > 0 else 0.0)
            fprs.append(fp / n_neg if n_neg > 0 else 0.0)

        fprs.append(1.0)
        tprs.append(1.0)
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        auc = np.trapezoid(tprs, fprs)

        results[g] = {"fpr": fprs, "tpr": tprs, "auc": float(auc)}
    return results


class AUCParity(FairnessMetric):
    """
    AUC Parity.

    Measures the difference in ROC AUC scores between groups.
    A value of 0 indicates perfect AUC parity.

    For 2 groups:
        value = AUC(S=a) - AUC(S=b)

    For more than 2 groups, value is the max AUC gap across all group pairs.

    The result also contains per-group ROC curves in `extra['roc_curves']`.

    References
    ----------
    .. [1] Hand, D.J. (2009). Measuring classifier performance: a coherent
           alternative to the area under the ROC curve. Machine Learning,
           77(1), 103-123.
           https://doi.org/10.1007/s10994-009-5119-5

    .. [2] Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L.
           (2019). Nuanced metrics for measuring unintended bias with real
           data for text classification. In Companion Proceedings of the
           World Wide Web Conference (pp. 491-500).
           https://doi.org/10.1145/3308560.3317593
    """

    def __init__(self, statistical_test: bool = False) -> None:
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"proba"}

    @property
    def name(self) -> str:
        return "auc_parity"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_proba(y_pred, "y_pred")
        s = _resolve_sensitive(sensitive)
        roc_data = _roc_curve_per_group(y_true, y_pred, s)

        aucs = {g: d["auc"] for g, d in roc_data.items()}
        values = list(aucs.values())
        gap = max(values) - min(values)

        roc_curves = {g: {"fpr": d["fpr"], "tpr": d["tpr"]} for g, d in roc_data.items()}

        return MetricResult(
            name=self.name,
            value=gap,
            groups=aucs,
            extra={
                "abs_value": abs(gap),
                "roc_curves": roc_curves,
            },
        )


class CalibrationParity(FairnessMetric):
    """
    Calibration Parity.

    Measures whether predicted probabilities are equally well calibrated
    across groups. Calibration is assessed via the Expected Calibration
    Error (ECE) per group.

    A value of 0 indicates perfect calibration parity.

    value = max ECE gap across all group pairs.

    References
    ----------
    .. [1] Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger,
           K.Q. (2017). On fairness and calibration. Advances in Neural
           Information Processing Systems, 30.
           https://arxiv.org/abs/1709.02012

    .. [2] Naeini, M.P., Cooper, G., & Hauskrecht, M. (2015). Obtaining
           well calibrated probabilities using Bayesian binning into quantiles.
           In Proceedings of the AAAI Conference on Artificial Intelligence
           (Vol. 29, No. 1).
    """

    def __init__(
        self,
        n_bins: int = 10,
        statistical_test: bool = False,
    ) -> None:
        self.n_bins = n_bins
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"proba"}

    @property
    def name(self) -> str:
        return "calibration_parity"

    def _ece(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(self.n_bins):
            mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_score[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_proba(y_pred, "y_pred")
        s = _resolve_sensitive(sensitive)
        groups = np.unique(s)

        eces = {}
        for g in groups:
            mask = s == g
            eces[g] = self._ece(y_true[mask], y_pred[mask])

        values = list(eces.values())
        gap = max(values) - min(values)

        return MetricResult(
            name=self.name,
            value=gap,
            groups=eces,
            extra={"abs_value": abs(gap)},
        )


class BrierParity(FairnessMetric):
    """
    Brier Score Parity.

    Measures the difference in Brier scores between groups.
    The Brier score is the mean squared error between predicted
    probabilities and true labels.

    A value of 0 indicates perfect Brier score parity.
    Lower Brier score = better calibrated model.

    For 2 groups:
        value = BrierScore(S=a) - BrierScore(S=b)

    For more than 2 groups, value is the max Brier gap across all group pairs.

    References
    ----------
    .. [1] Brier, G.W. (1950). Verification of forecasts expressed in terms
           of probability. Monthly Weather Review, 78(1), 1-3.
           https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

    .. [2] Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger,
           K.Q. (2017). On fairness and calibration. Advances in Neural
           Information Processing Systems, 30.
           https://arxiv.org/abs/1709.02012
    """

    def __init__(self, statistical_test: bool = False) -> None:
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"proba"}

    @property
    def name(self) -> str:
        return "brier_parity"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_proba(y_pred, "y_pred")
        s = _resolve_sensitive(sensitive)
        groups = np.unique(s)

        briers = {}
        for g in groups:
            mask = s == g
            briers[g] = float(np.mean((y_true[mask] - y_pred[mask]) ** 2))

        values = list(briers.values())
        gap = max(values) - min(values)

        return MetricResult(
            name=self.name,
            value=gap,
            groups=briers,
            extra={"abs_value": abs(gap)},
        )
