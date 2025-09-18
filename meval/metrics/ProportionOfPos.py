from typing import Optional
import pandas as pd

from .ComparisonMetric import ComparisonMetric
from ..group_filter import GroupFilter


class ProportionOfPos(ComparisonMetric):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols],
            metric_name='p(y=1|G=g)',
            reference_class='self',
            needs_pos_and_neg=False,
            is_descriptive=False,
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> float:
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        return y_true.sum() / mask.sum()
