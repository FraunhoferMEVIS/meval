from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint

from ._metrics import accuracy
from .ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar, MetricWithAnalyticalCI, ThresholdedComparisonMetric, MaskLike
from ..group_filter import GroupFilter
from ..stats import variance_of_proportion


class Accuracy(ThresholdedComparisonMetric, MetricWithAnalyticalVar, MetricWithAnalyticalCI):
    def __init__(self, threshold: Optional[float] = None, test: bool = False):
        super().__init__(
            threshold=threshold,
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_cols],
            metric_name='Acc',
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
        return_ci: bool = False,
        return_var: bool = False
        ) -> float | tuple[float, float] | tuple[float, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true_np = np.asarray(self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True), dtype=bool)
        y_pred_np = np.asarray(self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True), dtype=bool)

        if (not return_ci) and (not return_var):
            return accuracy(y_true_np, y_pred_np)
        
        if (not return_ci) and return_var:
            out = self.get_variance(df, group_mask=mask, y_true=y_true_np, y_pred=y_pred_np, validate=validate, return_val=True)
            assert isinstance(out, tuple)
            val, var = out
            return val, var

        if return_ci and not return_var:
            out = self.get_ci(df, group_mask=mask, y_true=y_true_np, y_pred=y_pred_np, validate=validate, return_val=True)
            assert isinstance(out, tuple)
            val, ci = out
            assert isinstance(ci, tuple)
            return val, ci
        
        raise NotImplementedError


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
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        if y_pred is None:
            y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True)

        y_true_np = np.asarray(y_true, dtype=bool)
        y_pred_np = np.asarray(y_pred, dtype=bool)

        var = variance_of_proportion(numerator=int((y_pred_np == y_true_np).sum()), denominator=len(y_pred_np))

        if return_val:
            return accuracy(y_true_np, y_pred_np), var
        else:
            return var
        
    def get_ci(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        y_true: Optional[pd.Series | np.ndarray] = None,
        y_pred: Optional[pd.Series | np.ndarray] = None,
        y_pred_prob: Optional[pd.Series | np.ndarray] = None,
        return_val: Optional[bool] = False
        ) -> tuple[float, float] | tuple[float, tuple[float, float]]:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        if y_pred is None:
            y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True)

        y_true_np = np.asarray(y_true, dtype=bool)
        y_pred_np = np.asarray(y_pred, dtype=bool)

        ci: tuple[float, float]
        ci = proportion_confint(count=int((y_true_np == y_pred_np).sum()), nobs=len(y_true_np), alpha=1-ci_alpha, method="wilson") # type: ignore

        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1
        assert ci[0] <= ci[1]

        if return_val:
            return accuracy(y_true_np, y_pred_np), ci
        else:
            return ci 




