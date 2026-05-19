from .discrete import (
    DemographicParity,
    EqualOpportunity,
    EqualizedOdds,
    PredictiveParity,
)
from .continuous import (
    AUCParity,
    CalibrationParity,
    BrierParity,
)

METRICS_REGISTRY: dict[str, type] = {
    "demographic_parity": DemographicParity,
    "equal_opportunity": EqualOpportunity,
    "equalized_odds": EqualizedOdds,
    "predictive_parity": PredictiveParity,
    "auc_parity": AUCParity,
    "calibration_parity": CalibrationParity,
    "brier_parity": BrierParity,
}

LABELS_METRICS = {
    name: cls
    for name, cls in METRICS_REGISTRY.items()
    if "labels" in cls().requires
}

PROBA_METRICS = {
    name: cls
    for name, cls in METRICS_REGISTRY.items()
    if "proba" in cls().requires
}
