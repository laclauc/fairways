from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PreProcessingResult:
    """
    Result of a pre-processing mitigation step.

    Attributes
    ----------
    X : np.ndarray or None
        Transformed feature matrix. None if unchanged.
    y : np.ndarray or None
        Transformed labels. None if unchanged.
    weights : np.ndarray or None
        Sample weights. None if not applicable.
    extra : dict[str, Any]
        Any additional method-specific information.
    """
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    weights: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = []
        if self.X is not None:
            parts.append(f"X={self.X.shape}")
        if self.y is not None:
            parts.append(f"y={self.y.shape}")
        if self.weights is not None:
            parts.append(f"weights={self.weights.shape}")
        return f"PreProcessingResult({', '.join(parts)})"


class FairnessMitigator(ABC):
    """Base class for all fairness mitigators."""

    @property
    def name(self) -> str:
        """Mitigator name — defaults to lowercase class name."""
        return self.__class__.__name__.lower()


class PreProcessor(FairnessMitigator):
    """
    Base class for pre-processing fairness mitigators.

    Pre-processors modify training data before model fitting.
    They operate on features X, labels y, and sensitive attributes.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> "PreProcessor":
        """
        Fit the mitigator on training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Ground truth labels of shape (n_samples,).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        PreProcessor
            The fitted mitigator (self).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> PreProcessingResult:
        """
        Apply the mitigation to training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Ground truth labels of shape (n_samples,).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        PreProcessingResult
            Transformed data with X, y, weights and/or extra fields.
        """
        raise NotImplementedError

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> PreProcessingResult:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Ground truth labels.
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).

        Returns
        -------
        PreProcessingResult
            Transformed data.
        """
        return self.fit(X, y, sensitive, **kwargs).transform(X, y, sensitive, **kwargs)


class PostProcessor(FairnessMitigator):
    """
    Base class for post-processing fairness mitigators.

    Post-processors adjust model predictions after fitting.
    They operate on predictions and sensitive attributes only —
    no access to the original features X is required.
    """

    @abstractmethod
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> "PostProcessor":
        """
        Fit the mitigator on predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Model predictions (labels or probabilities).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        PostProcessor
            The fitted mitigator (self).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        y_pred: np.ndarray,
        sensitive: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> np.ndarray:
        """
        Adjust predictions to reduce bias.

        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions (labels or probabilities).
        sensitive : np.ndarray or list[np.ndarray]
            Sensitive attribute(s).
        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        np.ndarray
            Adjusted predictions.
        """
        raise NotImplementedError
