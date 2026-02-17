import os
# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
from meval.metrics import Accuracy
from meval import compare_groups
from meval.config import settings

def test_multitest():
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

    # fdr_bh should be less conservative than holm, so p-values should be smaller (or equal if they are the same) after fdr_bh correction compared to holm
    # In this case, they are equal. Can we construct a test case where they are different?
    assert all_metric_results_df1.loc['group1=A', 'Acc pval'].item() >= all_metric_results_df2.loc['group1=A', 'Acc pval'].item()


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.update(parallel=False)

    test_multitest()
    test_fdr_bh()