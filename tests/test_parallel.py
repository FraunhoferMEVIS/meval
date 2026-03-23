import pytest
import pandas as pd
from meval.metrics import MultiClassAUROC
from meval import compare_groups
from meval.config import settings

@pytest.mark.timeout(120)
def test_error_propagation():
    test_df = pd.DataFrame({
        'y_pred_prob_0': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'y_pred_prob_1': [0.4, 0.1, 0.5, 0.1, 0.2, 0.1],
        'y_pred_prob_2': [0.5, 0.7, 0.2, 0.5, 0.3, 0.3],
        'GT': [0, 0, 1, 1, 2, 2]  # 'GT' is not an expected col name, so an error will be raised below
    })    
    test_df = pd.concat([test_df, test_df], ignore_index=True)
    test_df.loc[:, "site"] = 6*["A"] + 6*["B"]

    with pytest.raises(AssertionError) as exc_info:
        all_metric_results_df, plot_groups = compare_groups(
            df=test_df,
            metrics=[MultiClassAUROC()],
            group_by="site",
            min_subgroup_size=1
        )

    assert "Expected one of ['y_true', 'label', 'y', 'target'] in df.columns" in str(exc_info.value)

if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.load_testing_config(parallel=True)
    test_error_propagation()
