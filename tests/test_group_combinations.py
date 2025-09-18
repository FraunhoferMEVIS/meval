import pandas as pd
import pytest
from meval.compare_groups import get_group_combinations
import numpy as np


def unordered_lists_of_dicts_equal(lst1, lst2):

    set_a = {frozenset(d.items()) for d in lst1} 
    set_b = {frozenset(d.items()) for d in lst2}  

    return set_a == set_b   


def test_get_group_combinations():

    test_df = pd.DataFrame({
        "a": [0, 1, 2, 1],
        "b": ["x", "x", "y", "y"]
    })

    a_groups = get_group_combinations(test_df, ["a"], 0)
    assert unordered_lists_of_dicts_equal(a_groups, [{"a": v} for v in test_df["a"].unique()])

    b_groups = get_group_combinations(test_df, ["b"], 0)
    assert unordered_lists_of_dicts_equal(b_groups, [{"b": v} for v in test_df["b"].unique()])

    ab_groups = get_group_combinations(test_df, ["a", "b"], 1)
    assert unordered_lists_of_dicts_equal(ab_groups, [{"a": 0},
                              {"a": 1},
                              {"a": 2},
                              {"b": "x"},
                              {"b": "y"},
                              {"a": 0, "b": "x"},
                              {"a": 0, "b": "y"},
                              {"a": 1, "b": "x"},
                              {"a": 1, "b": "y"},
                              {"a": 2, "b": "x"},
                              {"a": 2, "b": "y"}])

    with pytest.raises(AssertionError):
        ab_groups = get_group_combinations(test_df, ["a", "b"], 2)


def test_get_group_combinations_with_NA():
    # pd.NA and np.nan really are different and might be in the dataframe for different reasons
    # https://stackoverflow.com/questions/60115806/pd-na-vs-np-nan-for-pandas
    # Silently merging these seems potentially dangerous, so let's keep them as they are
    test_df = pd.DataFrame({
        "a": [0, 1, pd.NA, 1],
        "b": ["x", "x", np.nan, np.nan]
    })

    a_groups = get_group_combinations(test_df, ["a"], 0)
    assert unordered_lists_of_dicts_equal(a_groups, [{"a": v} for v in test_df["a"].unique()])

    b_groups = get_group_combinations(test_df, ["b"], 0)
    assert unordered_lists_of_dicts_equal(b_groups, [{"b": v} for v in test_df["b"].unique()])

    ab_groups = get_group_combinations(test_df, ["a", "b"], 1)
    assert unordered_lists_of_dicts_equal(ab_groups, [{"a": 0},
                                {"a": 1},
                                {"a": pd.NA},
                                {"b": "x"},
                                {"b": np.nan},
                                {"a": 0, "b": "x"},
                                {"a": 0, "b": np.nan},
                                {"a": 1, "b": "x"},
                                {"a": 1, "b": np.nan},
                                {"a": pd.NA, "b": "x"},
                                {"a": pd.NA, "b": np.nan}])

    with pytest.raises(AssertionError):
        ab_groups = get_group_combinations(test_df, ["a", "b"], 2)


if __name__ == '__main__':
    test_get_group_combinations()
    test_get_group_combinations_with_NA()