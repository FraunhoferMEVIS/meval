from typing import Optional
import pandas as pd

from ._metrics import specificity
from .ComparisonMetric import ComparisonMetric, ThresholdedComparisonMetric
from ..group_filter import GroupFilter


class Specificity(ThresholdedComparisonMetric):
    # = 1 - FPR

    def __init__(self, threshold: Optional[float] = None, test: bool = False):

        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_cols],
            metric_name='Spec',
            threshold=threshold,
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
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate)

        return specificity(y_true.to_numpy(), y_pred.to_numpy())