import numpy as np
import pytest
from equiml.mitigation.pre.reweighting import Reweighting
from equiml.mitigation.base import PreProcessingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Simple binary dataset with group disparity."""
    np.random.seed(42)
    n = 200
    sensitive = np.array([0] * 100 + [1] * 100)
    y = np.array([1] * 60 + [0] * 40 + [1] * 30 + [0] * 70)
    X = np.random.randn(n, 4)
    return X, y, sensitive


@pytest.fixture
def balanced_data():
    """Dataset where groups and labels are independent."""
    sensitive = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y         = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X         = np.random.randn(8, 2)
    return X, y, sensitive


@pytest.fixture
def intersectional_data():
    """Dataset with 2 sensitive attributes."""
    np.random.seed(0)
    n = 200
    gender = np.array([0] * 100 + [1] * 100)
    age    = np.random.randint(0, 2, size=n)
    y      = np.array([1] * 60 + [0] * 40 + [1] * 30 + [0] * 70)
    X      = np.random.randn(n, 4)
    return X, y, [gender, age]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestReweightingFit:

    def test_returns_self(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        result = rw.fit(X, y, sensitive)
        assert result is rw

    def test_fitted_flag(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        assert not rw._fitted
        rw.fit(X, y, sensitive)
        assert rw._fitted

    def test_weights_map_populated(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        assert len(rw._weights_map) > 0

    def test_weights_map_keys(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        # Should have (group, label) pairs
        for key in rw._weights_map:
            assert len(key) == 2


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TestReweightingTransform:

    def test_returns_preprocessing_result(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        assert isinstance(result, PreProcessingResult)

    def test_X_unchanged(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        np.testing.assert_array_equal(result.X, X)

    def test_y_unchanged(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        np.testing.assert_array_equal(result.y, y)

    def test_weights_shape(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        assert result.weights.shape == (len(y),)

    def test_weights_positive(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        assert np.all(result.weights > 0)

    def test_weights_map_in_extra(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        rw.fit(X, y, sensitive)
        result = rw.transform(X, y, sensitive)
        assert "weights_map" in result.extra

    def test_transform_before_fit_raises(self, simple_data):
        X, y, sensitive = simple_data
        rw = Reweighting()
        with pytest.raises(RuntimeError, match="not fitted"):
            rw.transform(X, y, sensitive)


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------

class TestReweightingFitTransform:

    def test_fit_transform_equivalent(self, simple_data):
        X, y, sensitive = simple_data
        rw1 = Reweighting()
        rw1.fit(X, y, sensitive)
        result1 = rw1.transform(X, y, sensitive)

        rw2 = Reweighting()
        result2 = rw2.fit_transform(X, y, sensitive)

        np.testing.assert_array_almost_equal(result1.weights, result2.weights)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestReweightingCorrectness:

    def test_balanced_data_weights_near_one(self, balanced_data):
        """When S and Y are independent, weights should be close to 1."""
        X, y, sensitive = balanced_data
        rw = Reweighting()
        result = rw.fit_transform(X, y, sensitive)
        np.testing.assert_array_almost_equal(
            result.weights, np.ones(len(y)), decimal=5
        )

    def test_weighted_positive_rates_equal(self, simple_data):
        """After reweighting, weighted positive rates should be equal across groups."""
        X, y, sensitive = simple_data
        rw = Reweighting()
        result = rw.fit_transform(X, y, sensitive)

        groups = np.unique(sensitive)
        weighted_rates = []
        for g in groups:
            mask = sensitive == g
            w = result.weights[mask]
            weighted_rate = np.average(y[mask], weights=w)
            weighted_rates.append(weighted_rate)

        assert abs(weighted_rates[0] - weighted_rates[1]) < 1e-6

    def test_disadvantaged_group_gets_higher_weights(self, simple_data):
        """Positive examples from the disadvantaged group should get higher weights."""
        X, y, sensitive = simple_data
        rw = Reweighting()
        result = rw.fit_transform(X, y, sensitive)

        # Group 1 has fewer positives → positive examples should be upweighted
        w_pos_g0 = result.weights[(sensitive == 0) & (y == 1)].mean()
        w_pos_g1 = result.weights[(sensitive == 1) & (y == 1)].mean()
        assert w_pos_g1 > w_pos_g0


# ---------------------------------------------------------------------------
# Intersectional
# ---------------------------------------------------------------------------

class TestReweightingIntersectional:

    def test_intersectional_fit_transform(self, intersectional_data):
        X, y, sensitive = intersectional_data
        rw = Reweighting()
        result = rw.fit_transform(X, y, sensitive)
        assert isinstance(result, PreProcessingResult)
        assert result.weights.shape == (len(y),)
        assert np.all(result.weights > 0)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------

class TestReweightingName:

    def test_name(self):
        assert Reweighting().name == "reweighting"
