import numpy as np
import pandas as pd


def make_retail_data(
    periods: int = 365,
    seed: int = 7,
) -> pd.DataFrame:
    """Create synthetic daily retail demand with trend, seasonality, and promotions."""
    rng = np.random.default_rng(seed)
    day = np.arange(periods)
    weekly = 18 * np.sin(2 * np.pi * day / 7)
    annual = 28 * np.sin(2 * np.pi * (day - 20) / 365)
    trend = 0.12 * day

    promotion = (day % 47 == 0).astype(float)
    promotion_effect = promotion * 35
    event_noise = rng.normal(0.0, 7.0 + 4.0 * promotion, periods)
    demand = 120 + trend + weekly + annual + promotion_effect + event_noise
    demand = np.maximum(demand, 1)

    return pd.DataFrame({
        "day": day,
        "demand": demand,
        "promotion": promotion.astype(bool),
    })
