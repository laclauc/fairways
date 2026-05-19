import numpy as np
import pytest
from equiml.detection import detect_pred_type


class TestDetectPredType:

    # --- Integer dtype → labels ---

    def test_integer_array_is_labels(self):
        y = np.array([0, 1, 1, 0, 1], dtype=int)
        assert detect_pred_type(y) == "labels"

    def test_integer_array_int32_is_labels(self):
        y = np.array([0, 1, 1, 0], dtype=np.int32)
        assert detect_pred_type(y) == "labels"

    def test_integer_array_int64_is_labels(self):
        y = np.array([0, 1, 0, 1], dtype=np.int64)
        assert detect_pred_type(y) == "labels"

    # --- Float {0.0, 1.0} only → labels ---

    def test_float_binary_values_is_labels(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        assert detect_pred_type(y) == "labels"

    # --- Float in (0, 1) → proba ---

    def test_float_proba_is_proba(self):
        y = np.array([0.1, 0.8, 0.6, 0.3])
        assert detect_pred_type(y) == "proba"

    def test_float_proba_with_extremes_is_proba(self):
        # Contains 0.0 and 1.0 but also intermediate values
        y = np.array([0.0, 0.5, 1.0, 0.3])
        assert detect_pred_type(y) == "proba"

    # --- Out of range → error ---

    def test_float_above_one_raises(self):
        y = np.array([0.5, 1.5, 0.3])
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            detect_pred_type(y)

    def test_float_below_zero_raises(self):
        y = np.array([-0.1, 0.5, 0.8])
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            detect_pred_type(y)

    # --- Explicit pred_type override ---

    def test_explicit_labels_override(self):
        y = np.array([0.1, 0.8, 0.6])
        assert detect_pred_type(y, pred_type="labels") == "labels"

    def test_explicit_proba_override(self):
        y = np.array([0, 1, 0, 1])
        assert detect_pred_type(y, pred_type="proba") == "proba"

    def test_invalid_pred_type_raises(self):
        y = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="pred_type must be"):
            detect_pred_type(y, pred_type="invalid")

    # --- Unknown dtype → error ---

    def test_unknown_dtype_raises(self):
        y = np.array(["a", "b", "c"])
        with pytest.raises(ValueError):
            detect_pred_type(y)
