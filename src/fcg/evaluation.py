import numpy as np
import pandas as pd

from .forecasting import ForecastResult


def forecast_metrics(actual: np.ndarray, forecast: ForecastResult) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast.point)
    error = actual[valid] - forecast.point[valid]
    lower = forecast.lower[valid]
    upper = forecast.upper[valid]

    if not valid.any():
        return {"MAE": np.nan, "RMSE": np.nan, "coverage_95": np.nan, "mean_uncertainty": np.nan}

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "coverage_95": float(np.mean((actual[valid] >= lower) & (actual[valid] <= upper))),
        "mean_uncertainty": float(np.mean(forecast.uncertainty[valid])),
    }


def correlation_by_period(
    actual: np.ndarray,
    forecast: ForecastResult,
    highlight_width: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    absolute_error = np.abs(actual - forecast.point)
    valid = np.isfinite(absolute_error) & np.isfinite(highlight_width)
    x = absolute_error[valid]
    y = highlight_width[valid]

    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")
    return {"pearson_error_width": pearson, "spearman_error_width": spearman}
