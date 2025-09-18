from typing import Sequence
import pandas as pd
from .metrics.ComparisonMetric import ComparisonMetric


def select_extreme_tested_groups(all_metric_results_df: pd.DataFrame, 
                                 metrics: Sequence[ComparisonMetric],
                                 n_groups: int,
                                 add_all_group: bool = True
                                 ) -> list[str]:
    
    if add_all_group:
        assert 'all' in all_metric_results_df.index

    df = all_metric_results_df.copy()

    test_metrics = [metric for metric in metrics if metric.test]
    non_test_metrics = [metric for metric in metrics if not metric.test]

    offset = 1 if add_all_group else 0

    test_rank_cols = {}

    if len(test_metrics) > 0:
        for metric in test_metrics:
            pval_rank = df[metric.metric_name_pval].rank(method="average", ascending=True)
            effect_size_rank = df[metric.metric_name_effect].abs().rank(method="average", ascending=False)
            df.loc[:, metric.metric_name + " joint rank"] = pval_rank + effect_size_rank
            test_rank_cols[metric.metric_name + " joint rank"] = True  # True = sort ascending / select small vals first

        groups = select_extreme_groups(
            all_metric_results_df=df,
            cols=test_rank_cols,
            n_groups=n_groups-offset,
            add_all_group=False,
            sort=False
        )

    else:
        groups = []

    if len(groups) < n_groups-offset:
        other_cols = {metric.metric_name_low: False for metric in non_test_metrics} | {metric.metric_name_high: True for metric in non_test_metrics}
        other_groups = select_extreme_groups(
            all_metric_results_df=df,
            cols=other_cols,
            n_groups=n_groups-offset-len(groups),
            add_all_group=False,
            sort=False,
            exclude=groups
        )

        groups += other_groups
        test_rank_cols |= other_cols

    groups = list(df.loc[groups, :].sort_values(by=[metric.metric_name for metric in metrics if not metric.is_descriptive], ascending=False).index)

    if add_all_group:
        groups.append('all')

    return groups


def select_extreme_groups(all_metric_results_df: pd.DataFrame, 
                          cols: dict[str, bool],
                          n_groups: int,
                          add_all_group: bool = True,
                          sort: bool = True,
                          exclude: Sequence[str] = []
                          ) -> list[str]:
    
    offset = 1 if (add_all_group and 'all' not in exclude) else 0

    top = 0

    any_non_nan_entry_mask = all_metric_results_df[cols.keys()].notna().any(axis=1) & ~(all_metric_results_df.index == 'all') & ~ all_metric_results_df.index.isin(exclude)
    n_groups_with_at_least_one_non_nan_entry = any_non_nan_entry_mask.sum()

    if n_groups_with_at_least_one_non_nan_entry < (n_groups - offset):
        groups = list(all_metric_results_df.index[any_non_nan_entry_mask])

    else:
        groups = []
        
        while len(groups) < (n_groups - offset):

            for col, ascending in cols.items():
                grp = all_metric_results_df[col].sort_values(ascending=ascending, na_position='last').index[top]
                if grp not in groups and grp not in exclude and not grp == 'all':
                    groups.append(grp)

                    if len(groups) == (n_groups - offset):
                        break

            top += 1

    if sort:
        groups = list(all_metric_results_df.loc[groups, :].sort_values(by=list(cols.keys()), ascending=list(cols.values())).index)

    if add_all_group and 'all' not in exclude:
        groups.append('all')

    return groups