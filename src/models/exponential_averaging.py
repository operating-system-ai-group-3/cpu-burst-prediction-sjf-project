"""Traditional burst estimate that gives recent bursts more influence."""

from src.models.base import BurstPredictor


class ExponentialAveragingPredictor(BurstPredictor):
    def __init__(self, alpha: float = 0.5, initial_estimate: float = 10.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if initial_estimate <= 0:
            raise ValueError("initial_estimate must be positive")
        self.alpha = alpha
        self.initial_estimate = initial_estimate

    def fit(self, X, y):
        # This method has no learned parameters; it follows the shared API.
        return self

    def predict_next(self, history: list) -> float:
        # Start with a default estimate, then update it with older-to-newer data.
        estimate = self.initial_estimate
        for observed in history:
            estimate = self.alpha * observed + (1.0 - self.alpha) * estimate
        return float(max(0.1, estimate))
