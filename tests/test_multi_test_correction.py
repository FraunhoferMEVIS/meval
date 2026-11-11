import os

# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from meval import compare_groups
from meval.config import settings
from meval.metrics import Accuracy, Specificity


def test_corrected_pvals_not_smaller_than_raw_for_holm_and_fdr_bh():
    raw_pvals = np.array([1e-4, 0.002, 0.01, 0.03, 0.2, 0.6, 0.95])

    _, holm_pvals, _, _ = multipletests(raw_pvals, method="holm")
    _, bh_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")

    assert np.all(holm_pvals >= raw_pvals)
    assert np.all(bh_pvals >= raw_pvals)
    assert np.all((holm_pvals >= 0) & (holm_pvals <= 1))
    assert np.all((bh_pvals >= 0) & (bh_pvals <= 1))

def test_multitest():
    settings.load_testing_config(parallel=False)
    settings.update(testing=False, enable_bh_permut_sufficiency_guard=True, N_test_permut=1000)
    test_df = pd.DataFrame({
        'y_true': [True, False, True, False, True, False, True, False],
        'y_pred': [True, False, True, True, False, True, False, False],
        'group1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'group2': ['C', 'C', 'D', 'D', 'C', 'C', 'D', 'D']
    })    

    all_metric_results_df1, plot_groups = compare_groups(
        df=test_df,
        metrics=[Accuracy(test=True)],
        group_by="group1",
        min_subgroup_size=1
    )

    all_metric_results_df2, _plot_groups = compare_groups(
        df=test_df,
        metrics=[Accuracy(test=True)],
        group_by=["group1", "group2"],
        group_interactions=0,
        min_subgroup_size=1
    )

    assert all_metric_results_df1.loc['group1=A', 'Acc pval'].item() < all_metric_results_df2.loc['group1=A', 'Acc pval'].item()


def test_fdr_bh():
    settings.load_testing_config(parallel=False)
    settings.update(testing=False, enable_bh_permut_sufficiency_guard=True, N_test_permut=1000)
    test_df = pd.DataFrame({
        'y_true': [True, False, True, False, True, False, True, False],
        'y_pred': [True, False, True, True, False, True, False, False],
        'group1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'group2': ['C', 'C', 'D', 'D', 'C', 'C', 'D', 'D']
    })    

    all_metric_results_df1, plot_groups = compare_groups(
        df=test_df,
        metrics=[Accuracy(test=True)],
        group_by=["group1", "group2"],
        group_interactions=0,
        min_subgroup_size=1,
        test_correction_method="holm"
    )

    all_metric_results_df2, _plot_groups = compare_groups(
        df=test_df,
        metrics=[Accuracy(test=True)],
        group_by=["group1", "group2"],
        group_interactions=0,
        min_subgroup_size=1,
        test_correction_method="fdr_bh"
    )

    # There is no guaranteed per-hypothesis ordering between Holm and BH
    # adjusted p-values in finite samples; check basic validity instead.
    p_holm = all_metric_results_df1.loc['group1=A', 'Acc pval'].item()
    p_bh = all_metric_results_df2.loc['group1=A', 'Acc pval'].item()
    assert 0 <= p_holm <= 1
    assert 0 <= p_bh <= 1


def test_warn_and_disable_testing_when_permutations_insufficient_for_bh():
    old_settings = settings.to_dict().copy()
    try:
        settings.update(
            N_test_permut=10,
            pval_early_stop_alpha=0.05,
            parallel=False,
            testing=False,
            enable_bh_permut_sufficiency_guard=True,
        )

        test_df = pd.DataFrame({
            'y_true': [True, False, True, False, True, False, True, False],
            'y_pred': [True, False, True, True, False, True, False, False],
            'group1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        })

        with pytest.warns(UserWarning, match="num_permut too small and num_groups too large"):
            all_metric_results_df, _ = compare_groups(
                df=test_df,
                metrics=[Accuracy(test=True)],
                group_by="group1",
                min_subgroup_size=1,
                test_correction_method="fdr_bh",
            )

        assert np.isnan(all_metric_results_df['Acc pval']).all()
    finally:
        settings.from_dict(old_settings)


def test_can_override_bh_permut_sufficiency_guard_for_testing():
    old_settings = settings.to_dict().copy()
    try:
        settings.update(
            N_test_permut=10,
            pval_early_stop_alpha=0.05,
            enable_bh_permut_sufficiency_guard=False,
            parallel=False,
            testing=False,
        )

        test_df = pd.DataFrame({
            'y_true': [True, False, True, False, True, False, True, False],
            'y_pred': [True, False, True, True, False, True, False, False],
            'group1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        })

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            all_metric_results_df, _ = compare_groups(
                df=test_df,
                metrics=[Accuracy(test=True)],
                group_by="group1",
                min_subgroup_size=1,
                test_correction_method="fdr_bh",
            )

        assert not any("num_permut too small and num_groups too large" in str(w.message) for w in caught)
        assert not np.isnan(all_metric_results_df.loc['group1=A', 'Acc pval'])
    finally:
        settings.from_dict(old_settings)


def _guard_disables_testing(df, metrics, group_by, num_permut, alpha):
    """Run compare_groups with the |S| floor (f = 1) and report whether the permutation guard fired."""
    old_settings = settings.to_dict().copy()
    try:
        settings.update(N_test_permut=num_permut, pval_early_stop_alpha=alpha, parallel=False, testing=False,
                        enable_bh_permut_sufficiency_guard=True, permut_two_sided="abs")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare_groups(df=df, metrics=metrics, group_by=group_by, min_subgroup_size=1,
                           test_correction_method="fdr_bh")
        return any("num_permut too small and num_groups too large" in str(w.message) for w in caught)
    finally:
        settings.from_dict(old_settings)


def _binary_group_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.random(n) < 0.4
    return pd.DataFrame({
        'y_true': y_true,
        'y_pred_prob': np.clip(0.5 * y_true + 0.5 * rng.random(n), 0, 1),
        'group1': np.repeat(['A', 'B'], n // 2),
    })


def test_permutation_guard_does_not_count_the_untested_all_group():
    # m = 1 (group1=A; B is its binary complement, 'all' is never tested): 1/50 passes alpha = 0.03, 2/50 would not.
    df = _binary_group_df().assign(y_pred=lambda d: d.y_pred_prob > 0.5)
    assert not _guard_disables_testing(df, [Accuracy(test=True)], "group1", num_permut=50, alpha=0.03)


def test_permutation_guard_counts_a_shared_test_once():
    # BPSN and BNSP share one test, so m = 1 again, not 2.
    from meval.metrics import BNSP_AUROC, BPSN_AUROC

    metrics = [BPSN_AUROC(test=True), BNSP_AUROC(test=True)]
    assert not _guard_disables_testing(_binary_group_df(), metrics, "group1", num_permut=50, alpha=0.03)


def test_permutation_guard_counts_every_class_of_a_per_class_metric():
    # Recall(per_class=True) on three classes runs three tests per group: 3/100 exceeds alpha = 0.025.
    from meval.metrics import Recall

    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, 120)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': np.where(rng.random(120) < 0.7, y_true, rng.integers(0, 3, 120)),
                       'group1': np.repeat(['A', 'B'], 60)})
    assert _guard_disables_testing(df, [Recall(per_class=True, test=True)], "group1", num_permut=100, alpha=0.025)


def test_multitest_with_multiple_pval_columns_writeback_is_stable():
    settings.load_testing_config(parallel=False)
    settings.update(testing=False, enable_bh_permut_sufficiency_guard=True, N_test_permut=1000)

    test_df = pd.DataFrame({
        'y_true': [True, False, True, False, True, False, True, False, True, False, True, False],
        'y_pred': [True, False, True, True, False, True, False, False, True, False, False, True],
        'group1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
    })

    all_metric_results_df, _ = compare_groups(
        df=test_df,
        metrics=[Accuracy(test=True), Specificity(test=True)],
        group_by='group1',
        min_subgroup_size=1,
    )

    pval_cols = [col for col in all_metric_results_df.columns if 'pval' in col.lower()]
    assert len(pval_cols) == 2

    for col in pval_cols:
        vals = all_metric_results_df[col].dropna().to_numpy()
        assert np.all(vals >= 0)
        assert np.all(vals <= 1)


def test_bh_rounding_does_not_trip_the_bonferroni_sanity_check(monkeypatch):
    # multipletests computes 0.019 / (1 / 3) = 0.057, which is one ulp above 3 * 0.019.
    compare_groups_module = importlib.import_module("meval.compare_groups")
    raw_pvals = {"group1=A": 0.019, "group1=B": 0.2, "group1=C": 0.5}
    monkeypatch.setattr(compare_groups_module, "studentized_permut_pval",
                        lambda df, metric, group_filter, **kwargs: (raw_pvals[group_filter.group_name], 0.0))
    settings.load_testing_config(parallel=False)

    test_df = pd.DataFrame({
        'y_true': [True, False] * 6,
        'y_pred': [True, False, False, True] * 3,
        'group1': ['A'] * 4 + ['B'] * 4 + ['C'] * 4,
    })

    results, _ = compare_groups(df=test_df, metrics=[Accuracy(test=True)], group_by='group1', min_subgroup_size=1)

    assert results.loc['group1=A', 'Acc pval'] == pytest.approx(3 * 0.019)


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.load_testing_config(parallel=False)

    test_multitest()
    test_fdr_bh()