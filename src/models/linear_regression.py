"""Linear regression using the most recent CPU bursts as input features."""

import numpy as np
from sklearn.linear_model import LinearRegression

from src.models.base import BurstPredictor


class LinearRegressionPredictor(BurstPredictor):
    def __init__(self, window_size: int = 3, default_estimate: float = 10.0):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.default_estimate = default_estimate
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, X, y):
        if len(X) > 0:
            # Each training row is a fixed-size window of earlier bursts.
            self.model.fit(X, y)
            self.is_fitted = True
        return self

    def predict_next(self, history: list) -> float:
        if not self.is_fitted or len(history) == 0:
            return self.default_estimate
        # Repeat the first observation when history is shorter than the window.
        if len(history) < self.window_size:
            pad = [history[0]] * (self.window_size - len(history))
            window = pad + list(history)
        else:
            window = list(history)[-self.window_size :]
        pred = self.model.predict(np.array(window, dtype=float).reshape(1, -1))[0]
        return float(max(0.1, pred))
