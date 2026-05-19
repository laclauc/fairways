import numpy as np
from .base import FairnessMetric, MetricResult


def _check_binary(y: np.ndarray, name: str) -> None:
    """Raise if array is not binary {0, 1}."""
    unique = np.unique(y)
    if not set(unique).issubset({0, 1}):
        raise ValueError(
            f"{name} must be binary (0 or 1), got unique values: {unique}."
        )


def _group_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> dict:
    """
    Compute per-group confusion matrix rates.

    Returns
    -------
    dict
        Keys are group values, values are dicts with keys:
        'tpr', 'fpr', 'precision', 'positive_rate', 'n'.
    """
    groups = np.unique(sensitive)
    rates = {}
    for g in groups:
        mask = sensitive == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fn = np.sum((yt == 1) & (yp == 0))

        rates[g] = {
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "positive_rate": np.mean(yp),
            "n": int(mask.sum()),
        }
    return rates


def _resolve_sensitive(
    sensitive: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """
    Resolve sensitive attribute(s) into a single array.

    For intersectional analysis (list of arrays), combine them
    into a single array of tuple-like string labels.
    """
    if isinstance(sensitive, list):
        stacked = np.column_stack([s.astype(str) for s in sensitive])
        return np.array(["_".join(row) for row in stacked])
    return sensitive


class DemographicParity(FairnessMetric):
    """
    Demographic Parity (Statistical Parity).

    Measures the difference in positive prediction rates between groups.
    A value of 0 indicates perfect demographic parity.

    For 2 groups:
        value = P(Ŷ=1 | S=a) - P(Ŷ=1 | S=b)

    For more than 2 groups, value is the max gap across all group pairs.

    References
    ----------
    .. [1] Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R.
           (2012). Fairness through awareness. In Proceedings of the 3rd
           Innovations in Theoretical Computer Science Conference (pp. 214-226).
           https://doi.org/10.1145/2090236.2090255

    .. [2] Calders, T., & Verwer, S. (2010). Three naive Bayes approaches
           for discrimination-free classification. Data Mining and Knowledge
           Discovery, 21(2), 277-292.
           https://doi.org/10.1007/s10618-010-0190-x
    """

    def __init__(self, statistical_test: bool = False) -> None:
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"labels"}

    @property
    def name(self) -> str:
        return "demographic_parity"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_binary(y_pred, "y_pred")
        s = _resolve_sensitive(sensitive)
        rates = _group_rates(y_true, y_pred, s)

        positive_rates = {g: r["positive_rate"] for g, r in rates.items()}
        values = list(positive_rates.values())
        gap = max(values) - min(values)

        return MetricResult(
            name=self.name,
            value=gap,
            groups=positive_rates,
            extra={"abs_value": abs(gap)},
        )


class EqualOpportunity(FairnessMetric):
    """
    Equal Opportunity.

    Measures the difference in True Positive Rates (TPR) between groups.
    A value of 0 indicates equal opportunity.

    For 2 groups:
        value = TPR(S=a) - TPR(S=b)

    For more than 2 groups, value is the max TPR gap across all group pairs.

    References
    ----------
    .. [1] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity
           in supervised learning. Advances in Neural Information Processing
           Systems, 29, 3315-3323.
           https://arxiv.org/abs/1610.02413
    """

    def __init__(self, statistical_test: bool = False) -> None:
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"labels"}

    @property
    def name(self) -> str:
        return "equal_opportunity"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_binary(y_pred, "y_pred")
        _check_binary(y_true, "y_true")
        s = _resolve_sensitive(sensitive)
        rates = _group_rates(y_true, y_pred, s)

        tprs = {g: r["tpr"] for g, r in rates.items()}
        values = list(tprs.values())
        gap = max(values) - min(values)

        return MetricResult(
            name=self.name,
            value=gap,
            groups=tprs,
            extra={"abs_value": abs(gap)},
        )


class EqualizedOdds(FairnessMetric):
    """
    Equalized Odds.

    Measures differences in both TPR and FPR between groups.
    Returns the max of TPR gap and FPR gap.

    A value of 0 indicates equalized odds.

    References
    ----------
    .. [1] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity
           in supervised learning. Advances in Neural Information Processing
           Systems, 29, 3315-3323.
           https://arxiv.org/abs/1610.02413
    """

    def __init__(self, statistical_test: bool = False) -> None:
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"labels"}

    @property
    def name(self) -> str:
        return "equalized_odds"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_binary(y_pred, "y_pred")
        _check_binary(y_true, "y_true")
        s = _resolve_sensitive(sensitive)
        rates = _group_rates(y_true, y_pred, s)

        tprs = {g: r["tpr"] for g, r in rates.items()}
        fprs = {g: r["fpr"] for g, r in rates.items()}

        tpr_gap = max(tprs.values()) - min(tprs.values())
        fpr_gap = max(fprs.values()) - min(fprs.values())
        gap = max(tpr_gap, fpr_gap)

        return MetricResult(
            name=self.name,
            value=gap,
            groups=tprs,
            extra={
                "abs_value": abs(gap),
                "tpr_gap": tpr_gap,
                "fpr_gap": fpr_gap,
                "tprs": tprs,
                "fprs": fprs,
            },
        )


class PredictiveParity(FairnessMetric):
    """
    Predictive Parity.

    Measures the difference in precision between groups.
    A value of 0 indicates predictive parity.

    For 2 groups:
        value = Precision(S=a) - Precision(S=b)

    For more than 2 groups, value is the max precision gap.

    References
    ----------
    .. [1] Chouldechova, A. (2017). Fair prediction with disparate impact:
           A study of bias in recidivism prediction instruments. Big Data,
           5(2), 153-163.
           https://doi.org/10.1089/big.2016.0047
    """

    def __init__(self, statistical_test: bool = False) -> None:
        self.statistical_test = statistical_test

    @property
    def requires(self) -> set[str]:
        return {"labels"}

    @property
    def name(self) -> str:
        return "predictive_parity"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        _check_binary(y_pred, "y_pred")
        _check_binary(y_true, "y_true")
        s = _resolve_sensitive(sensitive)
        rates = _group_rates(y_true, y_pred, s)

        precisions = {g: r["precision"] for g, r in rates.items()}
        values = list(precisions.values())
        gap = max(values) - min(values)

        return MetricResult(
            name=self.name,
            value=gap,
            groups=precisions,
            extra={"abs_value": abs(gap)},
        )