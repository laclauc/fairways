import numpy as np
import pytest
from equiml.metrics.discrete import (
    DemographicParity,
    EqualOpportunity,
    EqualizedOdds,
    PredictiveParity,
    _resolve_sensitive,
)
from equiml.metrics.base import MetricResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Simple binary dataset with clear group disparity."""
    np.random.seed(42)
    y_true     = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_pred     = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    sensitive  = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return y_true, y_pred, sensitive


@pytest.fixture
def perfect_parity_data():
    """Dataset where both groups have identical prediction rates."""
    y_true    = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred    = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred, sensitive


@pytest.fixture
def multigroup_data():
    """Dataset with 3 sensitive groups."""
    y_true    = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred    = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return y_true, y_pred, sensitive


@pytest.fixture
def intersectional_data():
    """Dataset with 2 sensitive attributes for intersectional analysis."""
    y_true   = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred   = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    gender   = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    age      = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return y_true, y_pred, [gender, age]


# ---------------------------------------------------------------------------
# _resolve_sensitive
# ---------------------------------------------------------------------------

class TestResolveSensitive:

    def test_single_array_unchanged(self):
        s = np.array([0, 1, 0, 1])
        result = _resolve_sensitive(s)
        np.testing.assert_array_equal(result, s)

    def test_list_of_arrays_combined(self):
        s1 = np.array([0, 0, 1, 1])
        s2 = np.array([0, 1, 0, 1])
        result = _resolve_sensitive([s1, s2])
        expected = np.array(["0_0", "0_1", "1_0", "1_1"])
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# DemographicParity
# ---------------------------------------------------------------------------

class TestDemographicParity:

    def test_returns_metric_result(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert DemographicParity().name == "demographic_parity"

    def test_requires(self):
        assert DemographicParity().requires == {"labels"}

    def test_perfect_parity_is_zero(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert result.value == pytest.approx(0.0)

    def test_gap_is_non_negative(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert result.value >= 0.0

    def test_groups_in_result(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert 0 in result.groups
        assert 1 in result.groups

    def test_abs_value_in_extra(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert "abs_value" in result.extra
        assert result.extra["abs_value"] >= 0.0

    def test_multigroup(self, multigroup_data):
        y_true, y_pred, sensitive = multigroup_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 3

    def test_intersectional(self, intersectional_data):
        y_true, y_pred, sensitive = intersectional_data
        result = DemographicParity().compute(y_true, y_pred, sensitive)
        assert len(result.groups) == 4  # 2x2 groups

    def test_invalid_y_pred_raises(self, binary_data):
        y_true, _, sensitive = binary_data
        y_pred_invalid = np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        with pytest.raises(ValueError):
            DemographicParity().compute(y_true, y_pred_invalid, sensitive)


# ---------------------------------------------------------------------------
# EqualOpportunity
# ---------------------------------------------------------------------------

class TestEqualOpportunity:

    def test_returns_metric_result(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = EqualOpportunity().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert EqualOpportunity().name == "equal_opportunity"

    def test_requires(self):
        assert EqualOpportunity().requires == {"labels"}

    def test_perfect_parity_is_zero(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        result = EqualOpportunity().compute(y_true, y_pred, sensitive)
        assert result.value == pytest.approx(0.0)

    def test_gap_is_non_negative(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = EqualOpportunity().compute(y_true, y_pred, sensitive)
        assert result.value >= 0.0

    def test_groups_contain_tpr(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = EqualOpportunity().compute(y_true, y_pred, sensitive)
        for tpr in result.groups.values():
            assert 0.0 <= tpr <= 1.0


# ---------------------------------------------------------------------------
# EqualizedOdds
# ---------------------------------------------------------------------------

class TestEqualizedOdds:

    def test_returns_metric_result(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = EqualizedOdds().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert EqualizedOdds().name == "equalized_odds"

    def test_requires(self):
        assert EqualizedOdds().requires == {"labels"}

    def test_perfect_parity_is_zero(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        result = EqualizedOdds().compute(y_true, y_pred, sensitive)
        assert result.value == pytest.approx(0.0)

    def test_extra_contains_tpr_fpr_gaps(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = EqualizedOdds().compute(y_true, y_pred, sensitive)
        assert "tpr_gap" in result.extra
        assert "fpr_gap" in result.extra
        assert "tprs" in result.extra
        assert "fprs" in result.extra

    def test_value_is_max_of_tpr_fpr_gap(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = EqualizedOdds().compute(y_true, y_pred, sensitive)
        expected = max(result.extra["tpr_gap"], result.extra["fpr_gap"])
        assert result.value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# PredictiveParity
# ---------------------------------------------------------------------------

class TestPredictiveParity:

    def test_returns_metric_result(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = PredictiveParity().compute(y_true, y_pred, sensitive)
        assert isinstance(result, MetricResult)

    def test_name(self):
        assert PredictiveParity().name == "predictive_parity"

    def test_requires(self):
        assert PredictiveParity().requires == {"labels"}

    def test_perfect_parity_is_zero(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        result = PredictiveParity().compute(y_true, y_pred, sensitive)
        assert result.value == pytest.approx(0.0)

    def test_gap_is_non_negative(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = PredictiveParity().compute(y_true, y_pred, sensitive)
        assert result.value >= 0.0

    def test_groups_contain_precision(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = PredictiveParity().compute(y_true, y_pred, sensitive)
        for precision in result.groups.values():
            assert 0.0 <= precision <= 1.0
