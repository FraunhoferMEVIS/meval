from typing import Optional
import pandas as pd

from .ComparisonMetric import ComparisonMetric
from ..group_filter import GroupFilter


class Average(ComparisonMetric):

    def __init__(self, metric_col, test: bool = False):
        super().__init__(
            req_cols=[metric_col],
            metric_name="Avg(" + metric_col + ")",
            reference_class="self",
            needs_pos_and_neg=False,
            is_descriptive=False,
            test=test
        )
        self.metric_col = metric_col

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> float:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        return df[self.metric_col][mask].nanmean()
