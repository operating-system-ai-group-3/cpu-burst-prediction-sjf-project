"""Shared interface for all CPU burst prediction methods."""

from abc import ABC, abstractmethod


class BurstPredictor(ABC):
    """A small contract: train on examples, then predict from past bursts."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model using history windows ``X`` and next bursts ``y``."""
        raise NotImplementedError

    @abstractmethod
    def predict_next(self, history: list) -> float:
        """Estimate the next CPU burst using only the supplied history."""
        raise NotImplementedError
