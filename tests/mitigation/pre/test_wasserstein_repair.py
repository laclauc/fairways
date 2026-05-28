import numpy as np
import pytest
from equiml.mitigation.pre.wasserstein_repair import WassersteinRepair
from equiml.mitigation.base import PreProcessingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Two groups with different feature distributions."""
    np.random.seed(42)
    n = 200
    sensitive = np.array([0] * 100 + [1] * 100)
    # Group 0: mean=0, Group 1: mean=2 → clear disparity
    X = np.concatenate([
        np.random.normal(0, 1, (100, 3)),
        np.random.normal(2, 1, (100, 3)),
    ])
    y = np.random.randint(0, 2, n)
    return X, y, sensitive


@pytest.fixture
def identical_data():
    """Two groups with identical distributions."""
    np.random.seed(0)
    n = 200
    sensitive = np.array([0] * 100 + [1] * 100)
    X = np.random.normal(0, 1, (n, 3))
    y = np.random.randint(0, 2, n)
    return X, y, sensitive


@pytest.fixture
def single_feature_data():
    """1D feature array."""
    np.random.seed(0)
    sensitive = np.array([0] * 50 + [1] * 50)
    X = np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(3, 1, 50),
    ])
    y = np.random.randint(0, 2, 100)
    return X, y, sensitive


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestWassersteinRepairInit:

    def test_invalid_repair_type(self):
        with pytest.raises(ValueError, match="repair_type"):
            WassersteinRepair(repair_type="invalid")

    def test_invalid_lambda_above(self):
        with pytest.raises(ValueError, match="lambda_"):
            WassersteinRepair(lambda_=1.5)

    def test_invalid_lambda_below(self):
        with pytest.raises(ValueError, match="lambda_"):
            WassersteinRepair(lambda_=-0.1)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            WassersteinRepair(method="invalid")

    def test_valid_init(self):
        rw = WassersteinRepair(repair_type="geometric", lambda_=0.5)
        assert rw.repair_type == "geometric"
        assert rw.lambda_ == 0.5


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestWassersteinRepairFit:

    def test_returns_self(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair()
        assert rw.fit(X, y, sensitive) is rw

    def test_fitted_flag(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair()
        assert not rw._fitted
        rw.fit(X, y, sensitive)
        assert rw._fitted

    def test_groups_stored(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair()
        rw.fit(X, y, sensitive)
        assert len(rw._groups) == 2

    def test_more_than_two_groups_raises(self):
        X = np.random.randn(90, 3)
        y = np.random.randint(0, 2, 90)
        sensitive = np.array([0] * 30 + [1] * 30 + [2] * 30)
        rw = WassersteinRepair()
        with pytest.raises(ValueError, match="2 groups"):
            rw.fit(X, y, sensitive)

    def test_transform_before_fit_raises(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair()
        with pytest.raises(RuntimeError, match="not fitted"):
            rw.transform(X, y, sensitive)


# ---------------------------------------------------------------------------
# Transform — total repair
# ---------------------------------------------------------------------------

class TestWassersteinRepairTotal:

    def test_returns_preprocessing_result(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair(repair_type="total")
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        assert isinstance(result, PreProcessingResult)

    def test_y_unchanged(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)
        np.testing.assert_array_equal(result.y, y)

    def test_X_shape_preserved(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)
        assert result.X.shape == X.shape

    def test_group_means_closer_after_repair(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)

        # Before repair
        diff_before = abs(
            X[sensitive == 0].mean() - X[sensitive == 1].mean()
        )
        # After repair
        diff_after = abs(
            result.X[sensitive == 0].mean() - result.X[sensitive == 1].mean()
        )
        assert diff_after < diff_before

    def test_identical_groups_unchanged(self, identical_data):
        X, y, sensitive = identical_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)
        # Means should remain approximately equal
        assert abs(
            result.X[sensitive == 0].mean() - result.X[sensitive == 1].mean()
        ) < 0.5

    def test_extra_contains_repair_info(self, simple_data):
        X, y, sensitive = simple_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)
        assert "repair_type" in result.extra
        assert "lambda_" in result.extra
        assert "method" in result.extra


# ---------------------------------------------------------------------------
# Transform — geometric repair
# ---------------------------------------------------------------------------

class TestWassersteinRepairGeometric:

    def test_lambda_zero_unchanged(self, simple_data):
        """lambda_=0 should return the original data."""
        X, y, sensitive = simple_data
        rw = WassersteinRepair(repair_type="geometric", lambda_=0.0)
        result = rw.fit_transform(X, y, sensitive)
        np.testing.assert_array_almost_equal(result.X, X, decimal=10)

    def test_lambda_one_equals_total(self, simple_data):
        """lambda_=1 geometric should equal total repair."""
        X, y, sensitive = simple_data

        rw_geo = WassersteinRepair(repair_type="geometric", lambda_=1.0)
        result_geo = rw_geo.fit_transform(X, y, sensitive)

        rw_tot = WassersteinRepair(repair_type="total")
        result_tot = rw_tot.fit_transform(X, y, sensitive)

        np.testing.assert_array_almost_equal(
            result_geo.X, result_tot.X, decimal=8
        )

    def test_partial_repair_between_zero_and_one(self, simple_data):
        """lambda_=0.5 should be between no repair and full repair."""
        X, y, sensitive = simple_data

        rw = WassersteinRepair(repair_type="geometric", lambda_=0.5)
        result = rw.fit_transform(X, y, sensitive)

        diff_before = abs(
            X[sensitive == 0].mean() - X[sensitive == 1].mean()
        )
        diff_after = abs(
            result.X[sensitive == 0].mean() - result.X[sensitive == 1].mean()
        )
        # Should reduce disparity but not eliminate it
        assert diff_after < diff_before


# ---------------------------------------------------------------------------
# 1D feature
# ---------------------------------------------------------------------------

class TestWassersteinRepair1D:

    def test_1d_array(self, single_feature_data):
        X, y, sensitive = single_feature_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)
        assert result.X.shape == X.shape

    def test_1d_repair_reduces_disparity(self, single_feature_data):
        X, y, sensitive = single_feature_data
        rw = WassersteinRepair(repair_type="total")
        result = rw.fit_transform(X, y, sensitive)

        diff_before = abs(
            X[sensitive == 0].mean() - X[sensitive == 1].mean()
        )
        diff_after = abs(
            result.X[sensitive == 0].mean() - result.X[sensitive == 1].mean()
        )
        assert diff_after < diff_before


# ---------------------------------------------------------------------------
# OT method
# ---------------------------------------------------------------------------

class TestWassersteinRepairOT:

    def test_ot_not_implemented(self, simple_data):
        X, y, sensitive = simple_data
        try:
            rw = WassersteinRepair(method="ot")
            rw.fit(X, y, sensitive)
            with pytest.raises(NotImplementedError):
                rw.transform(X, y, sensitive)
        except ImportError:
            pytest.skip("POT not installed")


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------

class TestWassersteinRepairFitTransform:

    def test_fit_transform_equivalent(self, simple_data):
        X, y, sensitive = simple_data

        rw1 = WassersteinRepair(repair_type="geometric", lambda_=0.5)
        rw1.fit(X, y, sensitive)
        result1 = rw1.transform(X, y, sensitive)

        rw2 = WassersteinRepair(repair_type="geometric", lambda_=0.5)
        result2 = rw2.fit_transform(X, y, sensitive)

        np.testing.assert_array_almost_equal(result1.X, result2.X)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------

class TestWassersteinRepairName:

    def test_name(self):
        assert WassersteinRepair().name == "wasserstein_repair"
