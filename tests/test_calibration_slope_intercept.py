import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from meval import compare_groups
from meval.config import settings
from meval.diags import plot_metric_overview
from meval.metrics import AUROC, DRMSCE, CalibrationIntercept, CalibrationSlope, Count
from meval.metrics._calibration import fit_calibration_slope_intercept
from meval.stats.cache import clear_shared_computation_cache, shared_computation

CALIBRATION_SLOPE_MODULE = importlib.import_module("meval.metrics.CalibrationSlope")
CALIBRATION_INTERCEPT_MODULE = importlib.import_module("meval.metrics.CalibrationIntercept")


@pytest.fixture(autouse=True)
def _testing_config():
    settings.load_testing_config(parallel=False)
    clear_shared_computation_cache()


def _rng_df(n: int, seed: int, overconfidence: float = 1.0, group: bool | None = None) -> pd.DataFrame:
    """Synthetic binary df with y_pred_prob = sigmoid(overconfidence * true_logit).
    overconfidence=1.0 -> well calibrated; >1.0 -> overconfident (slope < 1)."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0, 1.2, n)
    p_true = 1 / (1 + np.exp(-true_logit))
    y_true = rng.binomial(1, p_true).astype(bool)
    y_pred_prob = 1 / (1 + np.exp(-overconfidence * true_logit))
    df = pd.DataFrame({"y_true": y_true, "y_pred_prob": y_pred_prob})
    if group is not None:
        df["group"] = group
    return df


def test_well_calibrated_data_gives_slope_near_1_intercept_near_0():
    df = _rng_df(n=2000, seed=0, overconfidence=1.0)

    results_df, _ = compare_groups(
        df=df, metrics=[CalibrationSlope(), CalibrationIntercept()], min_subgroup_size=1
    )

    assert results_df.loc["all", "Calibration slope"] == pytest.approx(1.0, abs=0.15)
    assert results_df.loc["all", "Calibration intercept"] == pytest.approx(0.0, abs=0.15)


def test_overconfident_data_gives_slope_below_1():
    df = _rng_df(n=2000, seed=1, overconfidence=2.5)

    results_df, _ = compare_groups(df=df, metrics=[CalibrationSlope()], min_subgroup_size=1)

    assert results_df.loc["all", "Calibration slope"] < 0.7


def test_too_small_group_returns_nan():
    rng = np.random.default_rng(2)
    # Only 3 positive cases, well below calibration_min_cases_per_class (default 10).
    y_true = np.array([True, True, True] + [False] * 20)
    y_pred_prob = rng.uniform(0.05, 0.95, len(y_true))
    df = pd.DataFrame({"y_true": y_true, "y_pred_prob": y_pred_prob})

    results_df, _ = compare_groups(
        df=df, metrics=[CalibrationSlope(), CalibrationIntercept()], min_subgroup_size=1
    )

    assert np.isnan(results_df.loc["all", "Calibration slope"])
    assert np.isnan(results_df.loc["all", "Calibration intercept"])


def test_fit_calibration_slope_intercept_returns_none_on_near_separation():
    # This case is not *perfectly* separated (statsmodels reports converged=True
    # and raises no PerfectSeparationWarning), but the fit is numerically
    # degenerate - huge, uninterpretable standard errors. This regression-tests
    # the MAX_INTERPRETABLE_SE guard specifically, not just the ordinary
    # separation exception/warning path.
    rng = np.random.RandomState(1)
    n = 15
    p_model = rng.uniform(0.01, 0.99, n)
    y_true = p_model > 0.5
    y_true[0] = not y_true[0]  # avoid *perfect* separation, near-separation remains

    fit = fit_calibration_slope_intercept(y_true, p_model, min_cases_per_class=2)

    assert fit is None


def test_shared_computation_caches_across_sibling_metrics(monkeypatch):
    """CalibrationSlope and CalibrationIntercept should reuse one fit per group,
    not refit once per metric."""
    df = _rng_df(n=300, seed=4)

    call_count = {"n": 0}
    original = fit_calibration_slope_intercept

    def _counting_fit(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(CALIBRATION_SLOPE_MODULE, "fit_calibration_slope_intercept", _counting_fit)
    monkeypatch.setattr(CALIBRATION_INTERCEPT_MODULE, "fit_calibration_slope_intercept", _counting_fit)

    compare_groups(df=df, metrics=[CalibrationSlope(), CalibrationIntercept()], min_subgroup_size=1)

    # Only the single 'all' group is analyzed here, so both metrics together
    # should trigger exactly one fit, not two.
    assert call_count["n"] == 1


def test_shared_computation_bypasses_cache_without_group_filter():
    # group_filter=None must NOT raise: this is the path significance testing
    # uses (permuted group_mask, no stable group identity), so it must work,
    # just without caching (each call recomputes).
    df = _rng_df(n=100, seed=5)
    call_count = {"n": 0}

    def compute():
        call_count["n"] += 1
        return 42

    result_1 = shared_computation("some_key", df, group_filter=None, compute_fn=compute)
    result_2 = shared_computation("some_key", df, group_filter=None, compute_fn=compute)

    assert result_1 == 42
    assert result_2 == 42
    assert call_count["n"] == 2  # not cached: recomputed both times


def test_calibration_slope_significance_testing_works_end_to_end():
    """test=True must work (not crash) and must be able to detect a real
    slope difference between groups when one is given enough permutations."""
    settings.update(N_test_permut=200, max_N_student_bootstrap=50)

    n_per_group = 600
    df_bad = _rng_df(n=n_per_group, seed=10, overconfidence=2.5, group="miscalibrated")
    df_good = _rng_df(n=n_per_group, seed=11, overconfidence=1.0, group="well_calibrated")
    df = pd.concat([df_bad, df_good], ignore_index=True)

    results_df, _ = compare_groups(
        df=df,
        metrics=[CalibrationSlope(test=True)],
        group_by="group",
        min_subgroup_size=1,
    )

    pval = results_df.loc["group=miscalibrated", "Calibration slope pval"]
    assert not np.isnan(pval)
    assert pval < 0.05


def test_curve_family_defaults_to_class_name_for_existing_metrics():
    # Existing metrics (not opted into a shared family) should keep their
    # original one-metric-one-plot behavior, keyed by their own class name.
    assert AUROC().curve_family == "AUROC"


def test_calibration_metrics_share_reliability_curve_family():
    assert DRMSCE().curve_family == "reliability"
    assert CalibrationIntercept().curve_family == "reliability"
    assert CalibrationSlope().curve_family == "reliability"


def test_plot_metric_overview_deduplicates_shared_curve_family():
    df = _rng_df(n=300, seed=6, group=True)
    df2 = _rng_df(n=300, seed=7, group=False)
    df = pd.concat([df, df2], ignore_index=True)

    metrics = [Count(), AUROC(), DRMSCE(), CalibrationIntercept(), CalibrationSlope()]
    results_df, plot_groups, metric_objs = compare_groups(
        df=df,
        metrics=metrics,
        group_by="group",
        min_subgroup_size=1,
        return_metric_objs=True,
    )

    fig = plot_metric_overview(metric_objs, results_df, plot_groups=plot_groups, test_df=df)

    axes_with_traces = {(trace.xaxis, trace.yaxis) for trace in fig.data}
    # 4 non-descriptive metric bar charts (AUROC, DRMSCE, Intercept, Slope)
    # + 2 curve plots (AUROC's own family, one shared "reliability" plot for
    # DRMSCE/Intercept/Slope combined) + 1 Count bar chart = 7, not 9 (which
    # would occur if each of DRMSCE/Intercept/Slope produced its own
    # reliability diagram).
    assert len(axes_with_traces) == 7


def test_compare_groups_integration_smoke_test():
    """End-to-end: a model that is overconfident in one group and well
    calibrated in another should show a group-level slope difference."""
    n_per_group = 400
    df_bad = _rng_df(n=n_per_group, seed=8, overconfidence=2.5, group="miscalibrated")
    df_good = _rng_df(n=n_per_group, seed=9, overconfidence=1.0, group="well_calibrated")
    df = pd.concat([df_bad, df_good], ignore_index=True)

    results_df, _ = compare_groups(
        df=df,
        metrics=[CalibrationSlope(), CalibrationIntercept()],
        group_by="group",
        min_subgroup_size=1,
    )

    slope_bad = results_df.loc["group=miscalibrated", "Calibration slope"]
    slope_good = results_df.loc["group=well_calibrated", "Calibration slope"]

    assert slope_bad < slope_good
    assert slope_good == pytest.approx(1.0, abs=0.25)


# --------------------------------------------------------------------------- #
# Reference-point (calibration bias) properties.
#
# The metric estimates Delta(L0) = a + (b - 1) * L0, the log-odds gap between
# the fitted truth and the model's own claim at the shared reference score L0.
# The properties below are the ones that make it usable for cross-group
# comparison: the estimand must depend only on the group's true calibration
# (a, b) and on L0, and never on the group's sample size, prevalence, or score
# distribution.
# --------------------------------------------------------------------------- #


def _biased_group_df(n, true_a, true_b, score_mean, score_sd, group, seed):
    """Group whose true calibration is logit(y) ~ true_a + true_b * logit(p),
    with its own score distribution N(score_mean, score_sd) on the logit scale.
    true_a = 0, true_b = 1 means perfectly calibrated."""
    rng = np.random.default_rng(seed)
    logit_p = rng.normal(score_mean, score_sd, n)
    y = rng.binomial(1, expit(true_a + true_b * logit_p)).astype(bool)
    return pd.DataFrame({"y_true": y, "y_pred_prob": expit(logit_p), "group": group})


def _bias_by_group(df, reference_score="mean", **kwargs):
    if df["group"].nunique() < 2:
        # compare_groups requires >1 level to stratify on. This filler group is
        # never asserted on; with a fixed reference_score it cannot influence
        # any other group's value (see the invariance test below).
        df = pd.concat([df, _biased_group_df(200, 0.0, 1.0, 0.0, 1.0, "_filler", seed=999)], ignore_index=True)
    results_df, _ = compare_groups(
        df=df,
        metrics=[CalibrationIntercept(reference_score=reference_score, **kwargs)],
        group_by="group",
        min_subgroup_size=1,
    )
    return results_df["Calibration intercept"]


def _replicate_diffs(n_a, n_b, mean_a, mean_b, true_b, true_gap, base_seed, n_rep=None):
    """Bias difference between groups A and B over independent replications.
    Group A's true calibration bias exceeds B's by exactly `true_gap`."""
    if n_rep is None:
        # Replications needed scale with the estimator's variance, which is
        # dominated by the smallest group and by how far its scores sit from
        # the shared reference point (the L0^2 * Var(b) term). Hard configs
        # therefore need more reps to pin down the mean.
        n_rep = 60 if (min(n_a, n_b) < 2000 or abs(mean_a - mean_b) > 3.0) else 12

    diffs = []
    for rep in range(n_rep):
        df = pd.concat(
            [
                _biased_group_df(n_a, 0.3 + true_gap, true_b, mean_a, 1.0, "A", seed=base_seed + 2 * rep),
                _biased_group_df(n_b, 0.3, true_b, mean_b, 1.0, "B", seed=base_seed + 2 * rep + 1),
            ],
            ignore_index=True,
        )
        bias = _bias_by_group(df)
        diffs.append(bias["group=A"] - bias["group=B"])
    diffs = np.asarray(diffs, dtype=float)
    return diffs[~np.isnan(diffs)]


@pytest.mark.parametrize(
    "n_a, n_b, mean_a, mean_b, true_b",
    [
        (4000, 4000, 0.0, 0.0, 1.0),      # matched everything, well-specified
        (4000, 4000, 0.0, -2.5, 1.0),     # very different prevalences
        (8000, 1000, 0.0, -2.5, 1.0),     # different prevalences AND sizes
        (4000, 4000, 0.0, -2.5, 1.8),     # different prevalences, slope != 1
        (8000, 1000, 1.5, -2.5, 1.8),     # everything differs at once
    ],
)
def test_no_difference_detected_when_none_exists(n_a, n_b, mean_a, mean_b, true_b):
    """(i) Groups sharing the same TRUE calibration must show no bias
    difference, regardless of their sample sizes, prevalences, or score
    spreads.

    Averaged over replications: the estimator is unbiased, but a *single*
    draw can be noisy when a small group's scores sit far from the shared
    reference point (the L0^2 * Var(b) term - see CalibrationIntercept's
    docstring). Unbiasedness is a property of the sampling distribution, so
    that is what is asserted here; per-draw spread is the job of the CIs.
    """
    diffs = _replicate_diffs(n_a, n_b, mean_a, mean_b, true_b, true_gap=0.0, base_seed=700)
    # Tolerance is ~2 standard errors of the noisiest configuration covered here.
    assert np.mean(diffs) == pytest.approx(0.0, abs=0.2)


@pytest.mark.parametrize(
    "n_a, n_b, mean_a, mean_b, true_b",
    [
        (4000, 4000, 0.0, 0.0, 1.0),
        (4000, 4000, 0.0, -2.5, 1.0),
        (8000, 1000, 0.0, -2.5, 1.0),
        (4000, 4000, 0.0, -2.5, 1.8),
    ],
)
def test_difference_detected_when_one_exists(n_a, n_b, mean_a, mean_b, true_b):
    """(ii) A real difference in true calibration bias must be recovered,
    regardless of sample sizes, prevalences, or score spreads. As above,
    asserted on the mean over replications."""
    true_gap = 0.8
    diffs = _replicate_diffs(n_a, n_b, mean_a, mean_b, true_b, true_gap=true_gap, base_seed=800)
    assert np.mean(diffs) == pytest.approx(true_gap, abs=0.2)
    # and the effect must be clearly separated from "no difference"
    assert np.mean(diffs) > 0.4


def test_estimate_is_unbiased_and_converges():
    """(iii) The estimator is asymptotically unbiased, and its finite-sample
    bias is small and shrinks with n.

    Note the maximum-likelihood estimates underlying this metric have a known
    O(1/n) small-sample bias (the reason Firth-type corrections exist), so this
    asserts that the bias is small and decreasing, not that it is exactly zero
    at finite n.
    """
    true_a, true_b = 0.5, 1.4
    # L0 is the dataset mean logit; with a single group at N(-1.0, 1.0) that is
    # about -1.0, so the target is Delta(L0) = a + (b - 1) * L0.
    # Use a FIXED reference point so the estimand is a constant across
    # replications (with reference_score="mean", L0 itself is a random
    # quantity, which would confound the bias measurement).
    reference_score = 0.2
    l0 = float(logit(reference_score))
    target = true_a + (true_b - 1.0) * l0

    errors = {}
    for n in (500, 5000, 50000):
        estimates = []
        for rep in range(12):
            df = _biased_group_df(n, true_a, true_b, -1.0, 1.0, "A", seed=1000 + rep + n)
            estimates.append(_bias_by_group(df, reference_score=reference_score)["group=A"] - target)
        errors[n] = float(np.nanmean(estimates))

    # Bias shrinks as n grows, and is negligible by n = 50k.
    assert abs(errors[50000]) < 0.02
    assert abs(errors[50000]) < abs(errors[500])


def test_reference_score_half_reproduces_plain_intercept():
    """reference_score=0.5 must reproduce the plain joint-fit intercept a,
    since Delta(0) = a. This is the documented backwards-compatible mode."""
    true_a = 0.4
    df = _biased_group_df(20000, true_a, 1.6, 0.0, 1.0, "A", seed=300)

    at_half = _bias_by_group(df, reference_score=0.5)["group=A"]
    assert at_half == pytest.approx(true_a, abs=0.1)


def test_reference_point_is_shared_across_groups_not_per_group():
    """The reference point must be a property of the whole dataset, so a
    group's reported bias must not change when an unrelated group's rows are
    present in the same call under a fixed reference."""
    group_a = _biased_group_df(4000, 0.3, 1.0, 0.0, 1.0, "A", seed=400)
    group_b = _biased_group_df(4000, 0.3, 1.0, -3.0, 1.0, "B", seed=401)

    # Under a FIXED reference, group A's value is invariant to B's presence.
    a_alone = _bias_by_group(group_a, reference_score=0.5)["group=A"]
    a_with_b = _bias_by_group(pd.concat([group_a, group_b], ignore_index=True), reference_score=0.5)["group=A"]
    assert a_alone == pytest.approx(a_with_b, abs=1e-6)


def test_invalid_reference_score_rejected():
    for bad in (0.0, 1.0, -0.5, 1.5, "median"):
        with pytest.raises(ValueError):
            CalibrationIntercept(reference_score=bad)
