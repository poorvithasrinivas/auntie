import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fcg.data import make_retail_data
from fcg.evaluation import correlation_by_period, forecast_metrics
from fcg.forecasting import gaussian_forecast, moving_average_forecast, residual_sharpness_forecast
from fcg.mapping import highlight_metrics
from fcg.rendering import render_specular_fallback


st.set_page_config(page_title="Forecast Sharpness via Phong Specular Exponent", layout="wide")
st.title("Forecast Sharpness: Phong n_s as Demand Uncertainty")
st.caption("A controlled visual-analytics study using one-step retail demand forecasts.")

with st.sidebar:
    periods = st.slider("Synthetic periods", 180, 730, 365, 30)
    window = st.slider("Forecast window", 3, 30, 7)
    noise = st.slider("Gaussian noise ratio", 0.05, 0.50, 0.20, 0.05)
    seed = st.number_input("Random seed", 0, 10000, 7)

data = make_retail_data(periods=int(periods), seed=int(seed))
demand = data["demand"].to_numpy()
naive = moving_average_forecast(demand, window=int(window))
gaussian = gaussian_forecast(demand, window=int(window), noise_std=float(noise), seed=int(seed))
proposed = residual_sharpness_forecast(demand, window=int(window))
metrics = highlight_metrics(proposed.uncertainty)

comparison = pd.DataFrame([
    {"Method": "Naive moving average", **forecast_metrics(demand, naive)},
    {"Method": "Gaussian noise", **forecast_metrics(demand, gaussian)},
    {"Method": "Proposed residual sharpness", **forecast_metrics(demand, proposed)},
])

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Demand and 95% forecast interval")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["day"], demand, label="Observed", color="#94a3b8", linewidth=1)
    ax.plot(data["day"], proposed.point, label="Point forecast", color="#38bdf8")
    ax.fill_between(data["day"], proposed.lower, proposed.upper, color="#38bdf8", alpha=0.22, label="95% interval")
    ax.set_xlabel("Day")
    ax.set_ylabel("Demand")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Forecast comparison")
    st.dataframe(comparison, hide_index=True)

st.subheader("Uncertainty maps to specular exponent")
plot_df = pd.DataFrame({
    "Day": data["day"],
    "Forecast uncertainty": proposed.uncertainty,
    "n_s": metrics.specular_exponent,
    "Highlight width": metrics.highlight_width,
})
st.line_chart(plot_df.set_index("Day")[["n_s", "Highlight width"]])

correlation = correlation_by_period(demand, proposed, metrics.highlight_width)
st.metric(
    "Pearson(error, highlight width)",
    f"{correlation['pearson_error_width']:.3f}",
)

c1, c2 = st.columns(2)
finite = pd.Series(proposed.uncertainty).dropna()
if not finite.empty:
    low_index = int(finite.idxmin())
    high_index = int(finite.idxmax())
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    low_path = output / "sharp_forecast_preview.png"
    high_path = output / "diffuse_forecast_preview.png"
    render_specular_fallback(float(metrics.specular_exponent[low_index]), low_path)
    render_specular_fallback(float(metrics.specular_exponent[high_index]), high_path)
    with c1:
        st.image(str(low_path), caption=f"High confidence preview: n_s = {metrics.specular_exponent[low_index]:.1f}")
    with c2:
        st.image(str(high_path), caption=f"Low confidence preview: n_s = {metrics.specular_exponent[high_index]:.1f}")
