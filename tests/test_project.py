import numpy as np

from fcg.evaluation import correlation_by_period
from fcg.forecasting import gaussian_forecast, moving_average_forecast, residual_sharpness_forecast
from fcg.mapping import highlight_metrics, specular_exponent


def test_moving_average_returns_forecasts():
    values = np.arange(1, 31, dtype=float)
    result = moving_average_forecast(values, window=7)
    assert np.isfinite(result.point[7:]).all()
    assert np.all(result.upper[7:] >= result.lower[7:])


def test_gaussian_forecast_has_uncertainty():
    values = np.arange(1, 31, dtype=float) ** 1.02
    result = gaussian_forecast(values, window=5)
    assert np.isfinite(result.uncertainty[5:].any())
    assert np.all(result.uncertainty[5:] > 0)


def test_specular_exponent_inverse_uncertainty():
    ns = specular_exponent(np.array([1.0, 2.0, 3.0]), ns_min=8, ns_max=200)
    assert ns[0] > ns[-1]
    assert ns.min() >= 8
    assert ns.max() <= 200


def test_highlight_width_monotonic_with_uncertainty():
    result = highlight_metrics(np.array([1.0, 2.0, 3.0]))
    assert result.highlight_width[0] < result.highlight_width[-1]


def test_error_width_correlation_is_valid():
    actual = np.array([10, 12, 13, 15])
    forecast = residual_sharpness_forecast(np.array([10, 11, 14, 16]), window=2)
    width = np.array([0.1, 0.2, 0.3, 0.4])
    correlations = correlation_by_period(actual, forecast, width)
    assert set(correlations) == {"pearson_error_width", "spearman_error_width"}
