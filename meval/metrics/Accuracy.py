from typing import Optional
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from ._metrics import accuracy
from .ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar, MetricWithAnalyticalCI, ThresholdedComparisonMetric
from ..group_filter import GroupFilter
from ..stats import variance_of_proportion


class Accuracy(ThresholdedComparisonMetric, MetricWithAnalyticalVar, MetricWithAnalyticalCI):
    def __init__(self, threshold: Optional[float] = None, test: bool = False):
        super().__init__(
            threshold=threshold,
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_cols],
            metric_name='Acc',
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
        validate: bool = True,
        return_ci: bool = False,
        return_var: bool = False
        ) -> float | tuple[float, float] | tuple[float, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate)

        if (not return_ci) and (not return_var):
            return accuracy(y_true.to_numpy(), y_pred.to_numpy())
        
        elif (not return_ci) and return_var:
            val, var = self.get_variance(df, group_mask=mask, y_true=y_true, y_pred=y_pred, validate=validate, return_val=True) # type: ignore
            return val, var

        elif return_ci and not return_var:
            val, ci = self.get_ci(df, group_mask=mask, y_true=y_true, y_pred=y_pred, validate=validate, return_val=True)
            return val, ci # type: ignore
        
        else:
            raise NotImplementedError


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

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        if y_pred is None:
            y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate)

        var = variance_of_proportion(numerator=(y_pred == y_true).sum(), denominator=len(y_pred))

        if return_val:
            return accuracy(y_true.to_numpy(), y_pred.to_numpy()), var
        else:
            return var
        
    def get_ci(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[pd.Series] = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
        y_pred_prob: Optional[pd.Series] = None,        
        return_val: Optional[bool] = False
        ) -> tuple[float, float] | tuple[float, tuple[float, float]]:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        if y_pred is None:
            y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate)

        ci: tuple[float, float]
        ci = proportion_confint(count=(y_true == y_pred).sum(), nobs=len(y_true), alpha=1-ci_alpha, method="wilson") # type: ignore

        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1
        assert ci[0] <= ci[1]

        if return_val:
            return accuracy(y_true.to_numpy(), y_pred.to_numpy()), ci
        else:
            return ci 