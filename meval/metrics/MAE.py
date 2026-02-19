from typing import Optional
import pandas as pd

from .ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar
from ..group_filter import GroupFilter


class MAE(MetricWithAnalyticalVar):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_float_pred_cols],
            metric_name='MAE',
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | tuple[float, float]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)

        y_true = self.get_float_y_true(df, mask=mask, validate=validate)
        y_pred = self.get_float_y_pred(df, mask=mask, validate=validate)

        abserrs = (y_true-y_pred).abs()
        mae = abserrs.mean()
        
        if return_var:
            var = abserrs.var(ddof=1) / len(abserrs)
            return mae, var
        else:
            return mae

    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
        y_pred_prob: Optional[pd.Series] = None,
        return_val: Optional[bool] = False
        ) -> float | tuple[float, float]:

        assert y_pred_prob is None, "MAE does not use y_pred_prob, expected y_pred_prob to be None."

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if y_true is None:
            y_true = self.get_float_y_true(df, mask=mask, validate=validate)
        if y_pred is None:
            y_pred = self.get_float_y_pred(df, mask=mask, validate=validate)

        abserrs = (y_true-y_pred).abs()

        var = abserrs.var(ddof=1) / len(abserrs)

        if return_val:
            return abserrs.mean(), var
        else:
            return var