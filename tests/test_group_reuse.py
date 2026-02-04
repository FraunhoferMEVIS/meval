import os
# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
from meval.metrics import BrierScore
from meval import compare_groups
from meval.config import settings


def test_integration():
    test_df = pd.DataFrame({
        'y_pred_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'y_true': [0, 0, 1, 1, 0, 1],
        'site': ['A', 'B', 'A', 'B', 'A', 'B'],
        'sex': ['M', 'M', 'M', 'F', 'F', 'F']
    })    
    test_df.y_true = test_df.y_true.astype(bool)

    all_metric_results_df, plot_groups = compare_groups(
        df=test_df,
        metrics=[BrierScore()],
        group_by=["site", "sex"],
        min_subgroup_size=1,
        group_interactions=1
    )

    # reuse groups from above both for analysis and plotting
    all_metric_results_df, plot_groups = compare_groups(
        df=test_df,
        metrics=[BrierScore()],
        analysis_groups=plot_groups,
        plot_groups=plot_groups
    )


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.update(N_bootstrap=10, N_test_permut=10, max_N_student_bootstrap=10, parallel=False)
    test_integration()
