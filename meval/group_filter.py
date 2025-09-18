from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa

numtypes = [int, float, np.floating, np.integer]

def safe_np_isnan(arr_or_series):
    # 1) pd.isna is True for both pd.NA and np.nan, but I want to distinguish between the two
    # 2) np.isnan throws errors for non-numeric series entries
    # This is essentially np.isnan except that it does not throw an error for non-numeric dtypes - 
    # for those, it simply returns False instead.
    # (It is also vectorized for efficiency.)

    if pd.api.types.is_numeric_dtype(arr_or_series.dtype):
        return np.isnan(arr_or_series)
    else:
        out = np.full(len(arr_or_series), False, dtype=bool)
        isnumeric = np.array([any([np.issubdtype(type(x), numtype) for numtype in numtypes]) for x in arr_or_series.values])
        out[isnumeric] = np.isnan(pd.to_numeric(arr_or_series[isnumeric], errors='raise'))
        return out


class GroupFilter(object):
    def __init__(
        self,
        group_attribute_dict: Optional[dict] = None,
        group_repr_str: Optional[str] = None,
        col_types: Optional[dict[str, npt.DTypeLike] | pd.Series] = None,
    ) -> None:
        # pass group_attribute_dict={} to create 'all' group
        if group_attribute_dict is not None:
            self.group_attributes = group_attribute_dict
        elif group_repr_str is not None:
            assert col_types is not None
            self.group_attributes = self.parse_attributes_from_repr(
                group_repr_str, col_types
            )
        else:
            raise ValueError

        self.predicate_length = len(self.group_attributes)

        # I'll probably want to introduce some abbreviation mechanism here at some point
        if self.predicate_length == 0:
            group_name = "all"
        else:
            group_name = ""
            for attribute in sorted(self.group_attributes.keys()):
                if len(group_name) > 0:
                    group_name = group_name + ", "
                group_name = (
                    group_name + f"{attribute}={self.group_attributes[attribute]}"
                )

        self.group_name = group_name

    @staticmethod
    def parse_attributes_from_repr(
        repr_str: str, col_types: dict[str, npt.DTypeLike] | pd.Series
    ) -> dict:
        if repr_str == "all":
            attribute_dict = {}
        else:
            preliminary_attribute_dict = dict(
                item.split("=") for item in repr_str.split(", ")
            )
            # the values are now all strings, convert to correct dtypes
            def parse_val(k, v):
                if v == "<NA>":
                    return pd.NA
                elif v == "nan":
                    return np.nan
                else:
                    return np.array(v, dtype=col_types[k]).item()

            attribute_dict = {
                k: parse_val(k, v)
                for k, v in preliminary_attribute_dict.items()
            }
        return attribute_dict
    
    @staticmethod
    def get_nan_type(v):
        # pd.isna is True for both pd.NA and np.nan. These are different however and may be in the df for different reasons,
        # so let's differentiate them.
        # In an earlier version, I did specifically check this using "v is pd.NA" and "v is np.nan", but 
        # that breaks in multiprocessing due to singleton serialization issues. Hence circumventing "is" below.
        try:
            if np.isnan(v):
                nantype = "np_nan"
            else:
                raise ValueError("Unknown NaN/NA type found?")
            
        except(TypeError):
            # np.isnan(pd.NA) yields <NA> (by pd NA propagation) and thus raises an error in the if clause above, 
            # as intended.
            # Verify this is really pd.NA and not something else that I don't know about.
            if type(v).__name__ == "NAType":
                nantype = "pd_NA"
            else:
                raise ValueError("Unknown NaN/NA type found?")
            
        return nantype

    def __call__(self, df: pd.DataFrame, validate: bool = True) -> pd.Series:
        filter_series = pd.Series(True, index=df.index, dtype=bool)

        for k, v in self.group_attributes.items():
            if pd.isna(v):
                nantype = self.get_nan_type(v)

                if nantype == "np_nan":
                    filter_series = filter_series & safe_np_isnan(df[k])
                elif nantype == "pd_NA":
                    filter_series = filter_series & (pd.isna(df[k]) & ~safe_np_isnan(df[k]))
                else:
                    raise ValueError("Unknown NaN/NA type found?")
                
            else:
                filter_series = filter_series & (df[k] == v)

        if validate:
            pa.SeriesSchema(bool, nullable=False, unique=False).validate(filter_series)

        return filter_series
    
    def complement(self, df: pd.DataFrame) -> pd.Series:
        # Filter for entries where none of the attributes matches the selected group's attributes
        # E.g., if the group is defined by gender=male and age=old, this will return a series
        # that is True for all entries that are neither male nor old.
        if self.predicate_length == 0:
            # 'all' group, there is no complement
            filter_series = pd.Series(False, index=df.index, dtype=bool)

        else:
            
            filter_series = pd.Series(True, index=df.index, dtype=bool)

            for k, v in self.group_attributes.items():

                if pd.isna(v):
                    nantype = self.get_nan_type(v)

                    if nantype == "np_nan":
                        filter_series = filter_series & ~safe_np_isnan(df[k])
                    elif nantype == "pd_NA":
                        filter_series = filter_series & ~(pd.isna(df[k]) & ~safe_np_isnan(df[k]))
                    else:
                        raise ValueError("Unknown NaN/NA type found?")                 
                       
                else:                
                    filter_series = filter_series & (df[k] != v)

        pa.SeriesSchema(bool, nullable=False, unique=False).validate(filter_series)

        return filter_series

    def __repr__(self) -> str:
        return "GroupFilter: " + self.group_name


def find_binary_complements(df, group_by, analysis_group_filters):
    # see explanation in compare_groups.py
    
    is_binary_attr = {}
    for attr in group_by:
        is_binary_attr[attr] = (len(df[attr].unique()) == 2)

    def is_all_binary_group(filter):
        if filter.predicate_length > 0:
            return all([is_binary_attr[attr] for attr in filter.group_attributes.keys()])
        else:
            # 'all' group which has no complement
            return False
    
    def binary_complement(group_filter):
        complement_attr_dict = {}
        for attr, val in group_filter.group_attributes.items():
            complement_attr_dict[attr] = [other_val for other_val in df[attr].unique() if not val == other_val].pop()

        complement_filter = GroupFilter(complement_attr_dict)
        assert all(group_filter.complement(df) == complement_filter(df))
        return complement_filter.group_name

    test_groups = []
    test_group_complements = {}

    for group_filter in analysis_group_filters:
        if is_all_binary_group(group_filter):
            complement_name = binary_complement(group_filter)
            if complement_name in test_groups:
                test_group_complements[group_filter.group_name] = complement_name
            else:
                test_groups.append(group_filter.group_name)
        else:
            test_groups.append(group_filter.group_name)    

    return test_groups, test_group_complements