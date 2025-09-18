from meval.group_filter import GroupFilter
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import repeat

def test_names_and_reprs():
    attr_dict = {'a': 2, 'b': 2.5, 'c': np.nan, 'd': pd.NA, 'e': 'yes', 'f': True}
    df = pd.DataFrame({k: [v] for k, v in attr_dict.items()})
    filter = GroupFilter(attr_dict)
    assert filter.group_name == "a=2, b=2.5, c=nan, d=<NA>, e=yes, f=True"

    attr_dict_rev = GroupFilter.parse_attributes_from_repr(filter.group_name, col_types=dict(df.dtypes))
    # The following fails (mistakenly) because pd.NA != pd.NA
    #assert attr_dict == attr_dict_rev
    
    for k, v in attr_dict.items():
        if v is pd.NA:
            assert attr_dict_rev[k] is pd.NA
        elif v is np.nan:
            assert attr_dict_rev[k] is np.nan
        else:
            assert v == attr_dict_rev[k]

    attr_dict_rev = GroupFilter.parse_attributes_from_repr("site=nan", col_types={'site': 'O'})  # object = str
    assert attr_dict_rev == {'site': np.nan}

    attr_dict_rev = GroupFilter.parse_attributes_from_repr("site=<NA>", col_types={'site': 'O'})  # object = str
    assert attr_dict_rev == {'site': pd.NA}


def apply_filter(filter, df):
    return filter(df)


def test_calls_with_nans():
    attr_dict = {'a': 2, 'b': 2.5, 'c': np.nan, 'd': pd.NA, 'e': 'yes', 'f': True}
    df = pd.DataFrame({k: [v] for k, v in attr_dict.items()})
    assert GroupFilter({'a': np.nan})(df).sum() == 0
    assert GroupFilter({'a': pd.NA})(df).sum() == 0
    assert GroupFilter({'a': 2})(df).sum() == 1      
    assert GroupFilter({'c': np.nan})(df).sum() == 1
    assert GroupFilter({'c': pd.NA})(df).sum() == 0
    assert GroupFilter({'d': np.nan})(df).sum() == 0
    assert GroupFilter({'d': pd.NA})(df).sum() == 1
    assert GroupFilter({'c': np.nan, 'd': pd.NA})(df).sum() == 1

    # Specifically test a case with mixed dtype
    df = pd.DataFrame({'a': [2, 'AA', np.nan], 'b': ['BB', 'CC', pd.NA]})
    assert GroupFilter({'a': 2})(df).sum() == 1
    assert GroupFilter({'a': np.nan})(df).sum() == 1
    assert GroupFilter({'a': pd.NA})(df).sum() == 0
    assert GroupFilter({'b': 'CC'})(df).sum() == 1
    assert GroupFilter({'b': np.nan})(df).sum() == 0
    assert GroupFilter({'b': pd.NA})(df).sum() == 1

    # Test that this also works in parallel (it did not before when I was using 'is pd.NA' and 'is np.nan')
    filters = [
        GroupFilter({'a': 2}),
        GroupFilter({'a': np.nan}),
        GroupFilter({'a': pd.NA})
    ]

    # This is loosely modeled after what we do in compare_groups.py
    with Pool(cpu_count()) as pool:
        masks_lst = pool.starmap(apply_filter, zip(filters, repeat(df)))

    assert all(masks_lst[0] == [True, False, False])
    assert all(masks_lst[1] == [False, False, True])
    assert all(masks_lst[2] == [False, False, False])


def test_complement():
    df = pd.DataFrame({'a': [2, 'AA', np.nan], 'b': ['BB', 'CC', pd.NA]})
    assert all(GroupFilter({'a': 2}).complement(df) == [False, True, True])
    assert all(GroupFilter({'a': np.nan}).complement(df) == [True, True, False])
    assert all(GroupFilter({'a': pd.NA}).complement(df) == [True, True, True])
    assert all(GroupFilter({'b': 'CC'}).complement(df) == [True, False, True])
    assert all(GroupFilter({'b': np.nan}).complement(df) == [True, True, True])
    assert all(GroupFilter({'b': pd.NA}).complement(df) == [True, True, False])


if __name__ == "__main__":
    test_names_and_reprs()
    test_calls_with_nans()
    test_complement()
