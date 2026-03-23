import os
# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
from meval.metrics import MultiClassAUROC
from meval import compare_groups
from meval.diags import metric_plot
from meval.config import settings


def test_multiauroc():
    test_df = pd.DataFrame({
        'y_pred_prob_0': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'y_pred_prob_1': [0.4, 0.1, 0.5, 0.1, 0.2, 0.1],
        'y_pred_prob_2': [0.5, 0.7, 0.2, 0.5, 0.3, 0.3],
        'y_true': [0, 0, 1, 1, 2, 2]
    })    
    test_df = pd.concat([test_df, test_df], ignore_index=True)
    test_df.loc[:, "site"] = 6*["A"] + 6*["B"]

    all_metric_results_df, plot_groups = compare_groups(
        df=test_df,
        metrics=[MultiClassAUROC()],
        group_by="site",
        min_subgroup_size=1
    )

    metric_plot(MultiClassAUROC(), all_metric_results_df)


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.load_testing_config(parallel=False)
    test_multiauroc()
