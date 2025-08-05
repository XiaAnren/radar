"""Metrics."""

import numpy as np
from scipy.ndimage import convolve


class FSSCalculator:
    """FSS (Fraction Skill Score) Calculator."""

    def __init__(self, threshold: int, radius: int) -> None:
        """FSS calculator with threshold and radius parameters.

        Args:
            threshold (int): Intensity value for converting fields to binary.
            radius (int): Neighborhood radius determining spatial scale.

        """
        window_size = 2 * radius + 1
        self.threshold = threshold
        self.kernel = np.ones((window_size, window_size)) / (window_size**2)

    def __call__(self, forecast: np.ndarray, observation: np.ndarray) -> float:
        """Calculate FSS between forecast and observation fields.

        Args:
            forecast (np.ndarray): 2D array of forecasted intensities.
            observation (np.ndarray): 2D array of observed intensities.

        Returns:
            float: FSS value between 0.0 (no skill) and 1.0 (perfect skill).

        """
        forecast = (forecast >= self.threshold).astype(float)
        observation = (observation >= self.threshold).astype(float)

        forecast = convolve(forecast, self.kernel, mode="constant")
        observation = convolve(observation, self.kernel, mode="constant")

        mse = np.mean((forecast - observation) ** 2)
        mse_ref = np.mean(forecast**2 + observation**2)

        if mse_ref == 0:
            return 1.0 if mse == 0 else 0.0
        return 1 - (mse / mse_ref)
