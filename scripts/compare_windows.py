"""Small sensitivity analysis over moving windows."""
from pathlib import Path

import pandas as pd

from fcg.data import make_retail_data
from fcg.evaluation import forecast_metrics
from fcg.forecasting import gaussian_forecast, moving_average_forecast, residual_sharpness_forecast


def main() -> None:
    data = make_retail_data(seed=7)
    demand = data["demand"].to_numpy()
    rows = []
    for window in (3, 7, 14, 30):
        forecasts = {
            "naive": moving_average_forecast(demand, window),
            "gaussian": gaussian_forecast(demand, window, seed=7),
            "proposed": residual_sharpness_forecast(demand, window),
        }
        for method, forecast in forecasts.items():
            rows.append({"window": window, "method": method, **forecast_metrics(demand, forecast)})

    output = Path("outputs")
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "window_sensitivity.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
