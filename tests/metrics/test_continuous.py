import numpy as np
import pytest
from equiml.metrics.continuous import (
    AUCParity,
    CalibrationParity,
    BrierParity,
)
from equiml.metrics.base import MetricResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def proba_data():
    """Simple dataset with clear group disparity in predicted probabilities."""
    np.random.seed(42)
    y_true    = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_pred    = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.1, 0.1,
                          0.8, 0.3, 0.6, 0.2])
    sensitive = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return y_true, y_pred, sensitive


@pytest.fixture
def perfect_parity_data():
    """Dataset where both groups have identical score distributions."""
    y_true    = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred    = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred, sensitive


@pytest.fixture
def multigroup_data():
    """Dataset with 3 sensitive groups."""
    y_true    = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred    = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.7, 0.3, 0.2,
                          0.9, 0.3, 0.1, 0.8])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return y_true, y_pred, sensitive


@pytest.fixture
def intersectional_data():
    """Dataset with 2 sensitive attributes for intersectional analysis."""
    y_true  = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred  = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.7, 0.3, 0.2])
    gender  = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    age     = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return y_true, y_pred, [gender, age]


# ---------------------------------------------------------------------------
# AUCParity
# ---------------------------------------------------------------------------

class TestAUCParity:

    def test_returns_metric_result(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert AUCParity().name == "auc_parity"

    def test_requires(self):
        assert AUCParity().requires == {"proba"}

    def test_perfect_parity_is_zero(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        assert result.value == pytest.approx(0.0, abs=1e-6)

    def test_gap_is_non_negative(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        assert result.value >= 0.0

    def test_auc_values_in_range(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        for auc in result.groups.values():
            assert 0.0 <= auc <= 1.0

    def test_roc_curves_in_extra(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        assert "roc_curves" in result.extra
        for g, curve in result.extra["roc_curves"].items():
            assert "fpr" in curve
            assert "tpr" in curve
            assert len(curve["fpr"]) == len(curve["tpr"])

    def test_multigroup(self, multigroup_data):
        y_true, y_pred, sensitive = multigroup_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 3

    def test_intersectional(self, intersectional_data):
        y_true, y_pred, sensitive = intersectional_data
        result = AUCParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 4

    def test_invalid_proba_raises(self, proba_data):
        y_true, _, sensitive = proba_data
        y_pred_invalid = np.array([1.5, 0.8, 0.4, 0.3, 0.7, 0.2,
                                   0.1, 0.1, 0.8, 0.3, 0.6, 0.2])
        with pytest.raises(ValueError):
            AUCParity().compute(y_true, y_pred_invalid, sensitive)


# ---------------------------------------------------------------------------
# CalibrationParity
# ---------------------------------------------------------------------------

class TestCalibrationParity:

    def test_returns_metric_result(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = CalibrationParity().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert CalibrationParity().name == "calibration_parity"

    def test_requires(self):
        assert CalibrationParity().requires == {"proba"}

    def test_gap_is_non_negative(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = CalibrationParity().compute(y_true, y_pred, sensitive)
        assert result.value >= 0.0

    def test_ece_values_in_range(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = CalibrationParity().compute(y_true, y_pred, sensitive)
        for ece in result.groups.values():
            assert 0.0 <= ece <= 1.0

    def test_custom_n_bins(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = CalibrationParity(n_bins=5).compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_multigroup(self, multigroup_data):
        y_true, y_pred, sensitive = multigroup_data
        result = CalibrationParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 3

    def test_intersectional(self, intersectional_data):
        y_true, y_pred, sensitive = intersectional_data
        result = CalibrationParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 4

    def test_invalid_proba_raises(self, proba_data):
        y_true, _, sensitive = proba_data
        y_pred_invalid = np.array([-0.1, 0.8, 0.4, 0.3, 0.7, 0.2,
                                    0.1, 0.1, 0.8, 0.3, 0.6, 0.2])
        with pytest.raises(ValueError):
            CalibrationParity().compute(y_true, y_pred_invalid, sensitive)


# ---------------------------------------------------------------------------
# BrierParity
# ---------------------------------------------------------------------------

class TestBrierParity:

    def test_returns_metric_result(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = BrierParity().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert BrierParity().name == "brier_parity"

    def test_requires(self):
        assert BrierParity().requires == {"proba"}

    def test_perfect_parity_is_zero(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        result = BrierParity().compute(y_true, y_pred, sensitive)
        assert result.value == pytest.approx(0.0, abs=1e-6)

    def test_gap_is_non_negative(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = BrierParity().compute(y_true, y_pred, sensitive)
        assert result.value >= 0.0

    def test_brier_values_in_range(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = BrierParity().compute(y_true, y_pred, sensitive)
        for brier in result.groups.values():
            assert 0.0 <= brier <= 1.0

    def test_multigroup(self, multigroup_data):
        y_true, y_pred, sensitive = multigroup_data
        result = BrierParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 3

    def test_intersectional(self, intersectional_data):
        y_true, y_pred, sensitive = intersectional_data
        result = BrierParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 4

    def test_invalid_proba_raises(self, proba_data):
        y_true, _, sensitive = proba_data
        y_pred_invalid = np.array([1.5, 0.8, 0.4, 0.3, 0.7, 0.2,
                                   0.1, 0.1, 0.8, 0.3, 0.6, 0.2])
        with pytest.raises(ValueError):
            BrierParity().compute(y_true, y_pred_invalid, sensitive)
