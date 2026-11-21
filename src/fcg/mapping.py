from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HighlightMetrics:
    specular_exponent: np.ndarray
    highlight_width: np.ndarray
    normalized_uncertainty: np.ndarray


def specular_exponent(
    uncertainty: np.ndarray,
    ns_min: float = 8.0,
    ns_max: float = 200.0,
    clip_percentiles: tuple[float, float] = (5.0, 95.0),
) -> np.ndarray:
    """Map low uncertainty to high Phong exponent and vice versa."""
    uncertainty = np.asarray(uncertainty, dtype=float)
    finite = uncertainty[np.isfinite(uncertainty)]
    if finite.size == 0:
        return np.full_like(uncertainty, (ns_min + ns_max) / 2)
    low, high = np.percentile(finite, clip_percentiles)
    scaled = np.clip((uncertainty - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
    return ns_max - scaled * (ns_max - ns_min)


def highlight_metrics(
    uncertainty: np.ndarray,
    ns_min: float = 8.0,
    ns_max: float = 200.0,
) -> HighlightMetrics:
    """Expose the rendered-width proxy and normalized uncertainty."""
    ns = specular_exponent(uncertainty, ns_min, ns_max)
    finite = uncertainty[np.isfinite(uncertainty)]
    if finite.size:
        low, high = np.percentile(finite, (5.0, 95.0))
        normalized = np.clip((uncertainty - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
    else:
        normalized = np.full_like(uncertainty, np.nan)

    # A monotonic, dimensionless proxy for angular highlight width.
    highlight_width = 1.0 / np.sqrt(np.maximum(ns, np.finfo(float).eps))
    return HighlightMetrics(ns, highlight_width, normalized)
