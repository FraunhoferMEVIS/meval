from typing import Optional
import numpy as np
import pandas as pd

from .ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar, MaskLike
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
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | tuple[float, float]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true_np = self.get_float_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred_np = self.get_float_y_pred(df, mask=mask, validate=validate, return_array=True)
        errs = np.abs(y_true_np - y_pred_np)
        n_total = errs.size
        n_valid = int(np.isfinite(errs).sum())

        if n_valid == 0:
            mae = np.nan
            var = np.nan
        else:
            mae = float(np.nanmean(errs))
            var = np.nan if n_valid <= 1 else float(np.nanvar(errs, ddof=1) / n_total)
        
        if return_var:
            return mae, var
        else:
            return mae

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

        assert y_pred_prob is None, "MAE does not use y_pred_prob, expected y_pred_prob to be None."

        mask: Optional[MaskLike]
        if y_true is None or y_pred is None:
            mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        else:
            mask = None

        if y_true is None:
            y_true_np = self.get_float_y_true(df, mask=mask, validate=validate, return_array=True)
        else:
            y_true_np = np.asarray(y_true, dtype=float)

        if y_pred is None:
            y_pred_np = self.get_float_y_pred(df, mask=mask, validate=validate, return_array=True)
        else:
            y_pred_np = np.asarray(y_pred, dtype=float)

        errs = np.abs(y_true_np - y_pred_np)
        n_valid = int(np.isfinite(errs).sum())

        if n_valid == 0:
            val = np.nan
            var = np.nan
        else:
            val = float(np.nanmean(errs))
            var = np.nan if n_valid <= 1 else float(np.nanvar(errs, ddof=1) / n_valid)

        if return_val:
            return val, var
        else:
            return var

