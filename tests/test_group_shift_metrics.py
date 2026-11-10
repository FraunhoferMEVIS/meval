"""Class-conditional averages, BPSN/BNSP and class-conditional AUROC, and the public prior-shift helper."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from meval import compare_groups
from meval.config import settings
from meval.metrics import AUROC, BNSP_AUROC, BPSN_AUROC, Average, ClassConditionalAUROC, logit_shift_correct
from meval.stats.bootstrap import bootstrap_metric
from meval.stats.studentized_permut_pval import est_variance_of_metric_diff


def _shortcut_frame(n=4000, seed=0):
    """Scores of group 'b' are shifted up by 1 logit unit regardless of the label."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b"], n)
    y = rng.binomial(1, 0.4, n).astype(bool)
    s = 1.5 * (2 * y - 1) + np.where(g == "b", 1.0, 0.0) + rng.normal(size=n)
    return pd.DataFrame({"y_true": y, "y_pred_prob": expit(s), "g": g})


def test_average_y_class_is_class_conditional_mean():
    df = _shortcut_frame()
    in_b = (df["g"] == "b").to_numpy()
    for label in (0, 1):
        expected = df.loc[in_b & (df["y_true"] == bool(label)), "y_pred_prob"].mean()
        metric = Average("y_pred_prob", y_class=label)
        assert metric(df, group_mask=in_b) == pytest.approx(expected)
        assert metric.metric_name == f"Avg(y_pred_prob | y={label})"
        # integer row positions (as used by resampling) select the same cases
        assert metric(df, group_mask=np.flatnonzero(in_b)) == pytest.approx(expected)


def test_average_without_y_class_is_unchanged():
    df = _shortcut_frame()
    assert Average("y_pred_prob")(df) == pytest.approx(df["y_pred_prob"].mean())


@pytest.mark.parametrize("metric_cls, group_class", [(BPSN_AUROC, False), (BNSP_AUROC, True)])
def test_background_auroc_matches_direct_computation(metric_cls, group_class):
    df = _shortcut_frame()
    in_b = (df["g"] == "b").to_numpy()
    y = df["y_true"].to_numpy()
    cases = (in_b & (y == group_class)) | (~in_b & (y != group_class))
    expected = roc_auc_score(y[cases], df["y_pred_prob"].to_numpy()[cases])

    metric = metric_cls()
    assert metric(df, group_mask=in_b) == pytest.approx(expected)
    value, (lo, hi) = metric(df, group_mask=in_b, return_ci=True)
    assert lo < value < hi
    assert value == pytest.approx(AUROC()(df, group_mask=cases))


def test_background_auroc_detects_group_shift():
    """Group b's scores are shifted up: its negatives look like positives (low BPSN) and its
    positives are easy to separate from the background's negatives (high BNSP)."""
    df = _shortcut_frame()
    in_b = (df["g"] == "b").to_numpy()
    within = AUROC()(df, group_mask=in_b)
    assert BPSN_AUROC()(df, group_mask=in_b) < within < BNSP_AUROC()(df, group_mask=in_b)


def test_bpsn_of_complement_is_bnsp_and_their_tests_coincide():
    """xAUC (Kallus & Zhou, 2019): BPSN(not G) = BNSP(G), so the group-vs-complement statistics
    of BPSN and BNSP are the same xAUC disparity with opposite signs."""
    df = _shortcut_frame(n=1000)
    in_b = (df["g"] == "b").to_numpy()
    assert BPSN_AUROC()(df, group_mask=~in_b) == pytest.approx(BNSP_AUROC()(df, group_mask=in_b))
    var_bpsn, diff_bpsn = est_variance_of_metric_diff(df, BPSN_AUROC(), in_b, ~in_b, max_num_bootstrap=0)
    var_bnsp, diff_bnsp = est_variance_of_metric_diff(df, BNSP_AUROC(), in_b, ~in_b, max_num_bootstrap=0)
    assert diff_bpsn == pytest.approx(-diff_bnsp)
    assert var_bpsn == pytest.approx(var_bnsp)
    within = AUROC()(df, group_mask=in_b)
    assert diff_bpsn < 0 < diff_bnsp and np.isfinite(within)


def test_background_auroc_ci_is_that_of_auroc_on_the_selected_cases():
    df = _shortcut_frame(n=1000)
    in_b = (df["g"] == "b").to_numpy()
    y = df["y_true"].to_numpy()
    cases = (in_b & ~y) | (~in_b & y)
    for method in ("delong", "newcombe"):
        assert BPSN_AUROC().get_ci(df, group_mask=in_b, method=method) == pytest.approx(
            AUROC().get_ci(df, group_mask=cases, method=method))
        assert BPSN_AUROC().get_variance(df, group_mask=in_b, method=method) == pytest.approx(
            AUROC().get_variance(df, group_mask=cases, method=method))
    assert BPSN_AUROC().get_ci(df, group_mask=in_b, method="delong") != pytest.approx(
        BPSN_AUROC().get_ci(df, group_mask=in_b, method="newcombe"))


def test_background_auroc_rejects_resampling_and_precomputed_arrays():
    df = _shortcut_frame(n=200)
    positions = np.flatnonzero((df["g"] == "b").to_numpy())
    assert BPSN_AUROC()(df, group_mask=positions) == pytest.approx(BPSN_AUROC()(df, group_mask=df["g"] == "b"))
    with pytest.raises(NotImplementedError):
        BPSN_AUROC()(df, group_mask=np.concatenate([positions, positions[:5]]))
    with pytest.raises(NotImplementedError):
        bootstrap_metric(df, BPSN_AUROC(), group_mask=positions, num_bootstrap=2)
    with pytest.raises(NotImplementedError):
        BPSN_AUROC().get_ci(df, group_mask=positions, y_true=df["y_true"], y_pred_prob=df["y_pred_prob"])
    assert np.isnan(BPSN_AUROC()(df))  # the 'all' group has no background


def test_class_conditional_auroc_special_cases():
    df = _shortcut_frame(n=1000)
    df["y_pred_prob"] = df["y_pred_prob"].round(1)  # ties count one half, as in the AUROC
    y = df["y_true"].to_numpy()
    in_b = (df["g"] == "b").to_numpy()
    for y_class in (0, 1):
        assert ClassConditionalAUROC(y_class)(df) == pytest.approx(AUROC()(df))
        assert ClassConditionalAUROC(y_class).metric_name == f"AUROC | y={y_class}"
    # a group without the other class: BPSN (y_class=0) or BNSP (y_class=1)
    negatives_of_b, positives_of_b = in_b & ~y, in_b & y
    assert ClassConditionalAUROC(0)(df, group_mask=negatives_of_b) == pytest.approx(
        BPSN_AUROC()(df, group_mask=negatives_of_b))
    assert ClassConditionalAUROC(1)(df, group_mask=positives_of_b) == pytest.approx(
        BNSP_AUROC()(df, group_mask=positives_of_b))
    # the shortcut: b's negatives score high (low value), b's positives score high (high value)
    overall = AUROC()(df)
    assert ClassConditionalAUROC(0)(df, group_mask=in_b) < overall < ClassConditionalAUROC(1)(df, group_mask=in_b)


def test_class_conditional_auroc_is_a_mean_of_placements():
    df = _shortcut_frame(n=500)
    in_b = (df["g"] == "b").to_numpy()
    for y_class in (0, 1):
        metric = ClassConditionalAUROC(y_class)
        (term,) = metric.decompose(df)
        value, var = metric(df, group_mask=in_b, return_var=True)
        selected = term.score[in_b & term.reference]
        assert value == pytest.approx(selected.mean())
        assert var == pytest.approx(selected.var(ddof=1) / len(selected))
        assert metric.get_variance(df, group_mask=in_b) == pytest.approx(var)
        # resampled groups (repeated positions) work: the reference is fixed by the dataset
        positions = np.flatnonzero(in_b)
        assert metric(df, group_mask=np.concatenate([positions, positions])) == pytest.approx(value)
        assert np.all(np.isfinite(bootstrap_metric(df, metric, group_mask=positions, num_bootstrap=5)))


def test_class_conditional_auroc_is_testable_without_the_other_class_in_the_group():
    df = _shortcut_frame(n=1000)
    no_positives = ((df["g"] == "b") & ~df["y_true"]).to_numpy()
    var, diff = est_variance_of_metric_diff(df, ClassConditionalAUROC(0), no_positives, ~no_positives,
                                            max_num_bootstrap=0)
    assert np.isfinite(var) and diff < 0
    var, diff = est_variance_of_metric_diff(df, BPSN_AUROC(), no_positives, ~no_positives, max_num_bootstrap=0)
    assert np.isnan(diff)


def _multiclass_frame(n=1500, n_classes=3, seed=1):
    """Group 'b' gets extra probability mass on class 2, regardless of the label."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b"], n)
    y = rng.integers(0, n_classes, n)
    logits = 1.5 * np.eye(n_classes)[y] + rng.normal(size=(n, n_classes))
    logits[:, 2] += np.where(g == "b", 1.0, 0.0)
    prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    df = pd.DataFrame({"y_true": y, "g": g})
    for c in range(n_classes):
        df[f"y_pred_prob_{c}"] = prob[:, c]
    return df


def _one_vs_rest_frame(df, class_id):
    return pd.DataFrame({"y_true": df["y_true"].to_numpy() == class_id,
                         "y_pred_prob": df[f"y_pred_prob_{class_id}"].to_numpy()})


@pytest.mark.parametrize("metric", [BPSN_AUROC(per_class=True), BNSP_AUROC(per_class=True),
                                    ClassConditionalAUROC(0, per_class=True), ClassConditionalAUROC(1, per_class=True)])
def test_per_class_metrics_are_one_vs_rest(metric):
    df = _multiclass_frame()
    in_b = (df["g"] == "b").to_numpy()
    resolved = metric.resolve_metrics(df)
    assert len(resolved) == 3
    binary_cls = type(metric)
    for class_id, per_class_metric in enumerate(resolved):
        binary = binary_cls(metric.y_class) if binary_cls is ClassConditionalAUROC else binary_cls()
        ovr = _one_vs_rest_frame(df, class_id)
        assert per_class_metric(df, group_mask=in_b) == pytest.approx(binary(ovr, group_mask=in_b))
        assert per_class_metric.get_variance(df, group_mask=in_b) == pytest.approx(
            binary.get_variance(ovr, group_mask=in_b))
    names = [m.metric_name for m in resolved]
    if binary_cls is ClassConditionalAUROC:
        sign = "=" if metric.y_class else "!="
        assert names == [f"AUROC[class={c}] | y{sign}{c}" for c in range(3)]
        assert resolved[1](df) == pytest.approx(roc_auc_score(df["y_true"] == 1, df["y_pred_prob_1"]))
    else:
        assert names == [f"{binary.metric_name}[class={c}]" for c in range(3)]


def test_per_class_flag_must_match_the_labels():
    with pytest.raises(NotImplementedError):
        BPSN_AUROC().resolve_metrics(_multiclass_frame(n=100))
    with pytest.raises(NotImplementedError):
        ClassConditionalAUROC(1, per_class=True).resolve_metrics(_shortcut_frame(n=100))


def test_multiclass_metrics_run_in_compare_groups():
    saved = settings.to_dict()
    settings.load_testing_config(parallel=False)
    try:
        results, _ = compare_groups(
            _multiclass_frame(n=600),
            metrics=[BPSN_AUROC(per_class=True), BNSP_AUROC(per_class=True, test=True),
                     ClassConditionalAUROC(0, per_class=True, test=True), ClassConditionalAUROC(1, per_class=True)],
            group_by=["g"],
        )
    finally:
        settings.from_dict(saved)
    assert "BNSP AUROC[class=2]" in results.columns
    assert "AUROC[class=2] | y!=2" in results.columns


def test_new_metrics_run_in_compare_groups():
    saved = settings.to_dict()
    settings.load_testing_config(parallel=False)
    try:
        df = _shortcut_frame(n=1000)
        results, _ = compare_groups(
            df,
            metrics=[BPSN_AUROC(test=True), BNSP_AUROC(), ClassConditionalAUROC(0, test=True),
                     ClassConditionalAUROC(1), Average("y_pred_prob", y_class=0), Average("y_pred_prob", y_class=1)],
            group_by=["g"],
        )
    finally:
        settings.from_dict(saved)
    assert len(results) > 0


def test_logit_shift_correct_is_public_and_shifts_logits():
    p = np.array([0.2, 0.5, 0.8])
    corrected = logit_shift_correct(p, pi_train=0.5, pi_test=0.9)
    assert np.allclose(np.log(corrected / (1 - corrected)) - np.log(p / (1 - p)), np.log(9.0))
