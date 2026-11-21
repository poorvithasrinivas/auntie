from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    uncertainty: np.ndarray


def moving_average_forecast(values: np.ndarray, window: int = 7) -> ForecastResult:
    """Return one-step forecasts and an empirical residual interval."""
    values = np.asarray(values, dtype=float)
    point = np.full(len(values), np.nan)
    residual_scale = np.full(len(values), np.nan)

    for index in range(window, len(values)):
        point[index] = values[index - window:index].mean()
        residual = values[window:index] - np.asarray([
            values[max(0, i - window):i].mean()
            for i in range(window, index)
        ])
        scale = np.std(residual, ddof=1) if len(residual) > 1 else 0.0
        residual_scale[index] = scale

    uncertainty = np.where(np.isnan(residual_scale), np.nan, residual_scale)
    lower = point - 1.96 * uncertainty
    upper = point + 1.96 * uncertainty
    return ForecastResult(point, lower, upper, uncertainty)


def gaussian_forecast(
    values: np.ndarray,
    window: int = 7,
    noise_std: float = 0.20,
    seed: int = 42,
) -> ForecastResult:
    """Jullien-style model: moving-average mean with injected Gaussian noise."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    point = np.full(len(values), np.nan)
    uncertainty = np.full(len(values), np.nan)

    for index in range(window, len(values)):
        baseline = values[index - window:index].mean()
        scale = noise_std * max(baseline, float(np.std(values[max(0, index-window):index])))
        samples = baseline + rng.normal(0.0, scale, size=1000)
        point[index] = samples.mean()
        uncertainty[index] = samples.std(ddof=1)

    lower = point - 1.96 * uncertainty
    upper = point + 1.96 * uncertainty
    return ForecastResult(point, lower, upper, uncertainty)


def residual_sharpness_forecast(values: np.ndarray, window: int = 7) -> ForecastResult:
    """Moving-average forecast whose uncertainty is estimated from recent errors."""
    values = np.asarray(values, dtype=float)
    point = np.full(len(values), np.nan)
    uncertainty = np.full(len(values), np.nan)

    for index in range(window, len(values)):
        point[index] = values[index - window:index].mean()
        start = max(window, index - window)
        prediction_errors = [
            values[j] - values[max(0, j-window):j].mean()
            for j in range(start, index)
        ]
        uncertainty[index] = np.std(prediction_errors, ddof=1) if len(prediction_errors) > 1 else 0.0

    lower = point - 1.96 * uncertainty
    upper = point + 1.96 * uncertainty
    return ForecastResult(point, lower, upper, uncertainty)
