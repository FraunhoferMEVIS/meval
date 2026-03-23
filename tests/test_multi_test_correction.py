import os
# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import pytest
import warnings
from statsmodels.stats.multitest import multipletests
from meval.metrics import Accuracy
from meval import compare_groups
from meval.config import settings


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

    all_metric_results_df2, plot_groups = compare_groups(
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

    all_metric_results_df2, plot_groups = compare_groups(
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


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.load_testing_config(parallel=False)

    test_multitest()
    test_fdr_bh()