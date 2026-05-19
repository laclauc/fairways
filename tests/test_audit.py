import numpy as np
import pytest
from equiml import audit
from equiml.result import AuditResult
from equiml.metrics.base import MetricResult
from equiml.metrics.discrete import DemographicParity, EqualizedOdds
from equiml.metrics.continuous import AUCParity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def label_data():
    y_true    = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_pred    = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return y_true, y_pred, sensitive


@pytest.fixture
def proba_data():
    y_true    = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_pred    = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2,
                          0.1, 0.1, 0.8, 0.3, 0.6, 0.2])
    sensitive = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return y_true, y_pred, sensitive


@pytest.fixture
def intersectional_data():
    y_true  = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred  = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    gender  = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    age     = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return y_true, y_pred, [gender, age]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestAuditReturnType:

    def test_returns_audit_result(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive)
        assert isinstance(result, AuditResult)

    def test_metrics_are_metric_results(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive)
        for v in result.metrics.values():
            assert isinstance(v, MetricResult)

    def test_groups_populated(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive)
        assert 0 in result.groups
        assert 1 in result.groups


# ---------------------------------------------------------------------------
# pred_type detection
# ---------------------------------------------------------------------------

class TestPredTypeDetection:

    def test_detects_labels(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive)
        assert result.pred_type == "labels"

    def test_detects_proba(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = audit(y_true, y_pred, sensitive)
        assert result.pred_type == "proba"

    def test_explicit_pred_type_override(self, label_data):
        y_true, y_pred, sensitive = label_data
        # Force proba even though values look like labels
        y_pred_float = y_pred.astype(float)
        result = audit(y_true, y_pred_float, sensitive, pred_type="labels")
        assert result.pred_type == "labels"


# ---------------------------------------------------------------------------
# metrics="all"
# ---------------------------------------------------------------------------

class TestMetricsAll:

    def test_all_labels_metrics_computed(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive, metrics="all")
        assert "demographic_parity" in result.metrics
        assert "equalized_odds" in result.metrics
        assert "equal_opportunity" in result.metrics
        assert "predictive_parity" in result.metrics

    def test_all_proba_metrics_computed(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = audit(y_true, y_pred, sensitive, metrics="all")
        assert "auc_parity" in result.metrics
        assert "calibration_parity" in result.metrics
        assert "brier_parity" in result.metrics

    def test_no_incompatible_metrics_with_labels(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive, metrics="all")
        assert "auc_parity" not in result.metrics

    def test_no_incompatible_metrics_with_proba(self, proba_data):
        y_true, y_pred, sensitive = proba_data
        result = audit(y_true, y_pred, sensitive, metrics="all")
        assert "demographic_parity" not in result.metrics


# ---------------------------------------------------------------------------
# metrics as list of strings
# ---------------------------------------------------------------------------

class TestMetricsStringList:

    def test_single_string_metric(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive,
                       metrics=["demographic_parity"])
        assert "demographic_parity" in result.metrics
        assert len(result.metrics) == 1

    def test_multiple_string_metrics(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive,
                       metrics=["demographic_parity", "equalized_odds"])
        assert "demographic_parity" in result.metrics
        assert "equalized_odds" in result.metrics

    def test_unknown_metric_raises(self, label_data):
        y_true, y_pred, sensitive = label_data
        with pytest.raises(ValueError, match="Unknown metric"):
            audit(y_true, y_pred, sensitive, metrics=["unknown_metric"])

    def test_incompatible_metric_raises(self, label_data):
        y_true, y_pred, sensitive = label_data
        with pytest.raises(ValueError):
            audit(y_true, y_pred, sensitive, metrics=["auc_parity"])


# ---------------------------------------------------------------------------
# metrics as list of objects
# ---------------------------------------------------------------------------

class TestMetricsObjectList:

    def test_metric_object(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive,
                       metrics=[DemographicParity()])
        assert "demographic_parity" in result.metrics

    def test_mixed_string_and_object(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(y_true, y_pred, sensitive,
                       metrics=["equalized_odds", DemographicParity()])
        assert "demographic_parity" in result.metrics
        assert "equalized_odds" in result.metrics

    def test_invalid_type_raises(self, label_data):
        y_true, y_pred, sensitive = label_data
        with pytest.raises(TypeError):
            audit(y_true, y_pred, sensitive, metrics=[42])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_mismatched_lengths_raises(self, label_data):
        y_true, y_pred, sensitive = label_data
        with pytest.raises(ValueError):
            audit(y_true, y_pred[:-1], sensitive)

    def test_mismatched_sensitive_length_raises(self, label_data):
        y_true, y_pred, sensitive = label_data
        with pytest.raises(ValueError):
            audit(y_true, y_pred, sensitive[:-1])

    def test_accepts_lists_as_input(self, label_data):
        y_true, y_pred, sensitive = label_data
        result = audit(
            y_true.tolist(),
            y_pred.tolist(),
            sensitive.tolist()
        )
        assert isinstance(result, AuditResult)


# ---------------------------------------------------------------------------
# Intersectional
# ---------------------------------------------------------------------------

class TestIntersectional:

    def test_intersectional_groups(self, intersectional_data):
        y_true, y_pred, sensitive = intersectional_data
        result = audit(y_true, y_pred, sensitive)
        assert len(result.groups) == 4

    def test_intersectional_metrics_computed(self, intersectional_data):
        y_true, y_pred, sensitive = intersectional_data
        result = audit(y_true, y_pred, sensitive,
                       metrics=["demographic_parity"])
        assert "demographic_parity" in result.metrics
