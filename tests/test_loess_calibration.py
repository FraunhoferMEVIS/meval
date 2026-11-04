import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from meval.config import settings
from meval.diags import rel_diag
from meval.metrics._calibration import loess_calibration, loess_span


def test_loess_span_shrinks_with_n():
    assert loess_span(100) == pytest.approx(0.75)
    assert loess_span(10_000) == pytest.approx(0.30, abs=0.005)
    spans = [loess_span(n) for n in (100, 1_000, 10_000, 100_000, 1_000_000)]
    assert all(a > b for a, b in zip(spans, spans[1:]))
    assert loess_span(10) == 1.0


def test_loess_span_setting_overrides_rule():
    settings.update(loess_frac=0.5)
    try:
        assert loess_span(100_000) == 0.5
    finally:
        settings.update(loess_frac=None)


def _miscalibrated_sample(n, rng):
    """Scores concentrated in [0.2, 0.6]; true curve logit P(y=1 | p) = 0.3 + 0.8 * logit(p)."""
    p = rng.uniform(0.2, 0.6, n)
    y = rng.binomial(1, expit(0.3 + 0.8 * logit(p)))
    return pd.Series(y), p


def test_loess_curve_is_nan_outside_observed_scores_and_inside_unit_interval():
    rng = np.random.default_rng(0)
    y, p = _miscalibrated_sample(5_000, rng)
    xvals = np.linspace(0, 1, 101)
    curve, samples = loess_calibration(y, p, n_bootstrap_samples=5, xvals=xvals)

    outside = (xvals < p.min()) | (xvals > p.max())
    assert np.all(np.isnan(curve[outside]))
    assert np.all(np.isnan(samples[outside]))
    assert np.all((curve[~outside] >= 0) & (curve[~outside] <= 1))

    truth = expit(0.3 + 0.8 * logit(xvals[~outside]))
    assert np.max(np.abs(curve[~outside] - truth)) < 0.05


def test_rel_diag_handles_partial_support():
    rng = np.random.default_rng(1)
    y, p = _miscalibrated_sample(2_000, rng)
    df = pd.DataFrame({"y_true": y.astype(bool), "y_pred_prob": p, "g": rng.choice(["a", "b"], len(p))})
    previous = settings.num_loess_calibration_samples
    settings.update(num_loess_calibration_samples=5)
    try:
        fig, traces, _ = rel_diag(df, plot_groups=["all", "g=a", "g=b"])
    finally:
        settings.update(num_loess_calibration_samples=previous)
    bands = [t for t in traces if t.fill == "toself"]
    assert len(bands) == 3
    for band in bands:
        assert not np.any(np.isnan(np.asarray(band.y, dtype=float)))
        assert np.min(band.x) >= p.min() - 1e-9 and np.max(band.x) <= p.max() + 1e-9
