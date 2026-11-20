"""Random forest regression model for next CPU burst prediction."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BurstPredictor


class RandomForestPredictor(BurstPredictor):
    def __init__(self, window_size: int = 3, n_estimators: int = 30, max_depth: int = 6, seed: int = 42, default_estimate: float = 10.0):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.default_estimate = default_estimate
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1
        )
        self.is_fitted = False

    def fit(self, X, y):
        if len(X) > 0:
            # A fixed random seed and one worker keep runs repeatable.
            self.model.fit(X, y)
            self.is_fitted = True
        return self

    def predict_next(self, history: list) -> float:
        if not self.is_fitted or len(history) == 0:
            return self.default_estimate
        # Make the live history match the feature width used during training.
        if len(history) < self.window_size:
            pad = [history[0]] * (self.window_size - len(history))
            window = pad + list(history)
        else:
            window = list(history)[-self.window_size :]
        pred = self.model.predict(np.array(window, dtype=float).reshape(1, -1))[0]
        return float(max(0.1, pred))
