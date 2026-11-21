# Forecast Sharpness as a Phong Specular Exponent

This project tests the hypothesis that a Phong specular exponent can encode confidence in a retail demand forecast:

- high `n_s` (narrow highlight) = sharp, high-confidence forecast;
- low `n_s` (wide/diffuse highlight) = uncertain forecast.

The implementation includes a synthetic retail experiment, naive and Gaussian-noise baselines, a residual-based proposed method, a Blender rendering script, a local fallback renderer, and a Streamlit dashboard.

## Visual mapping

For uncertainty `u`, the project maps the 5th--95th percentile of finite observations to:

```text
n_s = n_s_max - normalized_uncertainty * (n_s_max - n_s_min)
```

Default bounds are `n_s = 8` to `200`. A monotonic highlight-width proxy is `1 / sqrt(n_s)`. The Blender script maps `n_s` to Principled-BSDF roughness (`1 / sqrt(n_s)`) because Blender's modern shader does not expose the classic Phong exponent directly.

## Setup

```powershell
cd "C:\Users\poorv\OneDrive\Desktop\fcg mini\fcg_project"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the experiment:

```powershell
python scripts/run_experiment.py --seed 7 --output outputs
```

Launch the dashboard:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pip install pytest
pytest -q
```

## Optional Blender render

Install Blender, then run:

```powershell
blender --background --python scripts/blender_render.py -- 200.0 "outputs/blender_ns200.png"
blender --background --python scripts/blender_render.py -- 8.0 "outputs/blender_ns8.png"
```

The script writes the original `n_s` value into custom material metadata and stores a roughness approximation for rendering.

## Methods

**Naive moving average (Eq. 2):** each forecast is the mean of the preceding `w` observations.

**Gaussian injection:** a moving-average baseline is perturbed with Gaussian noise whose scale is derived from the baseline and recent demand variation. This represents an injected-noise baseline rather than a statistically calibrated forecast interval.

**Proposed residual sharpness:** the forecast remains a moving average, while uncertainty is estimated from recent one-step prediction residuals. This uncertainty is rescaled to `n_s`, so low uncertainty produces a sharp highlight and high uncertainty produces a diffuse highlight.

**Evaluation:** MAE and RMSE measure forecast accuracy; 95% interval coverage measures calibration; Pearson and Spearman correlations test whether highlight width varies with absolute forecast error.

## Publication-oriented study structure

For a Methods/Journal submission, report:

1. The synthetic-data generating process and random seed.
2. Window and noise parameter sensitivity.
3. Train/test or rolling-origin evaluation, not only descriptive dashboard results.
4. Baseline and proposed MAE, RMSE, 95% coverage, and correlation coefficients.
5. A preregistered null hypothesis: no monotonic relationship between highlight width and absolute error.
6. Replication across seeds and window sizes.
7. The limitation that the current experiment uses a rendering proxy and a synthetic dataset.
