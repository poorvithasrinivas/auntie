import argparse
from pathlib import Path

import pandas as pd

from fcg.data import make_retail_data
from fcg.evaluation import correlation_by_period, forecast_metrics
from fcg.forecasting import gaussian_forecast, moving_average_forecast, residual_sharpness_forecast
from fcg.mapping import highlight_metrics
from fcg.rendering import render_specular_fallback


def run(seed: int = 7, output_dir: Path = Path("outputs")) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_retail_data(seed=seed)
    demand = data["demand"].to_numpy()

    forecasts = {
        "naive_moving_average": moving_average_forecast(demand),
        "gaussian_noise": gaussian_forecast(demand, seed=seed),
        "proposed_residual": residual_sharpness_forecast(demand),
    }

    rows = []
    for name, forecast in forecasts.items():
        metrics = forecast_metrics(demand, forecast)
        if name == "proposed_residual":
            metrics.update(correlation_by_period(
                demand,
                forecast,
                highlight_metrics(forecast.uncertainty).highlight_width,
            ))
        rows.append({"method": name, **metrics})

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "experiment_metrics.csv", index=False)

    proposed = forecasts["proposed_residual"]
    metrics = highlight_metrics(proposed.uncertainty)
    representative = [
        int(pd.Series(proposed.uncertainty).idxmin()),
        int(pd.Series(proposed.uncertainty).idxmax()),
    ]
    for index in representative:
        if index < len(demand):
            render_specular_fallback(
                float(metrics.specular_exponent[index]),
                output_dir / f"highlight_{index}.png",
            )

    data.assign(
        predicted=proposed.point,
        lower_95=proposed.lower,
        upper_95=proposed.upper,
        uncertainty=proposed.uncertainty,
    ).to_csv(output_dir / "forecast_output.csv", index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(run(args.seed, args.output).to_string(index=False))
