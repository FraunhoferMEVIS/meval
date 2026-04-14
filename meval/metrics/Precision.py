from typing import Optional
import pandas as pd

from ._metrics import precision
from .ComparisonMetric import ComparisonMetric, ThresholdedComparisonMetric, MaskLike
from ..group_filter import GroupFilter


class Precision(ThresholdedComparisonMetric):

    def __init__(self, threshold: Optional[float] = None, test: bool = False):

        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_cols],
            metric_name='Prec',
            threshold=threshold,
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True)

        return precision(y_true, y_pred)




