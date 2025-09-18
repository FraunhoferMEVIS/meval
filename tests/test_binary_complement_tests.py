import os
# Add these BEFORE importing pandas/numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
from meval import compare_groups
from meval.metrics import BrierScore
from meval.group_filter import find_binary_complements, GroupFilter
from meval.compare_groups import get_group_combinations
from meval.config import settings

def setup():
    # a-c are binary attrs, d is not
    df = pd.DataFrame({'a': [2, 3, 2, 2, 2, 3, 2, 2, 3, 3, 2], 
                       'b': ['BB', 'CC', 'BB', 'CC', 'BB', 'BB', 'BB', 'CC', 'BB', 'BB', 'CC'], 
                       'c': [True, False, False, False, True, True, False, False, True, True, False], 
                       'd': ['a', 'b', pd.NA, 'a', 'b', pd.NA, 'a', 'b', pd.NA, 'a', 'b'], 
                       'y_true': [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
                       'y_pred_prob': [0.1, 0.0, 1.0, 0.3, 0.2, 0.5, 0.7, 0.9, 0.4, 0.2, 0.0]
    })
    df.y_true = df.y_true.astype(bool)

    metrics = [BrierScore(test=True, balanced=False)]

    group_by = ["a", "b", "c", "d"]

    analysis_groups = [{}] + get_group_combinations(df, group_by, max_combinations=3)
    analysis_group_filters = [GroupFilter(group_attribute_dict) for group_attribute_dict in analysis_groups]    

    return df, metrics, group_by, analysis_group_filters


def test_find_binary_complements():

    df, metrics, group_by, analysis_group_filters = setup()

    test_groups, test_group_complements = find_binary_complements(df, group_by=group_by,
                                                                  analysis_group_filters=analysis_group_filters)

    # There can be nothing involving the ternary group in the complements
    assert not any(['d' in groupname for groupname in test_group_complements.keys()])
    assert not any(['d' in groupname for groupname in test_group_complements.values()])

    assert 'a=2' in test_group_complements.keys() or 'a=2' in test_group_complements.values()
    assert "b=BB" in test_group_complements.keys() or "b=BB" in test_group_complements.values()
    assert "c=True" in test_group_complements.keys() or "c=True" in test_group_complements.values()

    # Specifically, we expecting the following complement pairs:
    # a=2 vs 3
    # b=BB vs CC
    # c=True vs False
    # all permutations of these three = a total of 3 + 6 + 4 = 13 entries
    assert len(test_group_complements.keys()) == 13
    assert all([k != v for k, v in test_group_complements.items()])
    assert all([k not in test_groups for k in test_group_complements.keys()])


def test_binary_complement_tests():

    df, metrics, group_by, _ = setup()

    results_df, _ = compare_groups(df, metrics, group_by=group_by, group_interactions=1, min_subgroup_size=1)

    # Single binary attribute
    assert results_df.loc['a=2', 'BS pval'] == results_df.loc['a=3', 'BS pval']
    assert results_df.loc['a=2', 'BS effect'] == -results_df.loc['a=3', 'BS effect']

    # Two binary attributes
    assert results_df.loc['a=2, b=CC', 'BS pval'] == results_df.loc['a=3, b=BB', 'BS pval']
    assert results_df.loc['a=2, b=CC', 'BS effect'] == -results_df.loc['a=3, b=BB', 'BS effect']


if __name__ == '__main__':
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None    

    settings.update(N_bootstrap=10, N_test_permut=10, max_N_student_bootstrap=10, parallel=False)
    test_find_binary_complements()
    test_binary_complement_tests()