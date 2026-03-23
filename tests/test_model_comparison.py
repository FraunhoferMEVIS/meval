import os
# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
from meval.metrics import BrierScore, Precision
from meval import compare_groups
from meval.config import settings


def test_comparison():
    test_df_1 = pd.DataFrame({
        'y_pred_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'y_true': [0, 0, 1, 1, 0, 1],
        'site': ['A', 'B', 'A', 'B', 'A', 'B'],
        'sex': ['M', 'M', 'M', 'F', 'F', 'F']
    })    
    test_df_1.y_true = test_df_1.y_true.astype(bool)

    test_df_2 = pd.DataFrame({
        'y_pred_prob': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        'y_true': [0, 0, 1, 1, 0, 1],
        'site': ['A', 'B', 'A', 'B', 'A', 'B'],
        'sex': ['M', 'M', 'M', 'F', 'F', 'F']
    })    
    test_df_2.y_true = test_df_2.y_true.astype(bool)    

    _, _ = compare_groups(
        df={'Model A': test_df_1, 'Model B': test_df_2},
        metrics=[BrierScore(), Precision()],
        group_by=["site", "sex"],
        min_subgroup_size=1,
        group_interactions=1
    )


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.load_testing_config(parallel=False)
    settings.update(N_bootstrap=5, N_test_permut=5, max_N_student_bootstrap=5)
    test_comparison()
