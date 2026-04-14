from typing import Optional
import pandas as pd
import numpy as np

from .ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar, MaskLike
from ..group_filter import GroupFilter

class Average(MetricWithAnalyticalVar):

    def __init__(self, metric_col, test: bool = False):
        super().__init__(
            req_cols=[metric_col],
            metric_name="Avg(" + metric_col + ")",
            reference_class="self",
            needs_all_classes=False,
            is_descriptive=False,
            test=test
        )
        self.metric_col = metric_col

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | tuple[float, float]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)

        vals = ComparisonMetric._masked_array(df[self.metric_col], mask)
        
        n_valid = np.isfinite(vals).sum()
        if n_valid == 0:
            avg, var = np.nan, np.nan
        else:
            avg = float(np.nanmean(vals))
            var = np.nan if n_valid <= 1 else float(np.nanvar(vals, ddof=1) / n_valid)
        
        return (avg, var) if return_var else avg


    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        y_true: Optional[pd.Series | np.ndarray] = None,
        y_pred: Optional[pd.Series | np.ndarray] = None,
        y_pred_prob: Optional[pd.Series | np.ndarray] = None,
        return_val: Optional[bool] = False
        ) -> float | tuple[float, float]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        vals = ComparisonMetric._masked_array(df[self.metric_col], mask)

        n_valid = np.isfinite(vals).sum()
        if n_valid == 0:
            val, var = np.nan, np.nan
        else:
            val = float(np.nanmean(vals))
            var = np.nan if n_valid <= 1 else float(np.nanvar(vals, ddof=1) / n_valid)

        return (val, var) if return_val else var