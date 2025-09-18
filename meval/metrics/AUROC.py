from typing import Optional
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go
import pandas as pd
from confidenceinterval.delong import delong_roc_variance
import numpy as np
import scipy.stats

from ._metrics import auroc
from .ComparisonMetric import ComparisonMetric, CurveBasedComparisonMetric, MetricWithAnalyticalCI, MetricWithAnalyticalVar
from ..diags import roc_diag
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups
from ..stats import newcombe_auroc_ci


class AUROC(CurveBasedComparisonMetric, MetricWithAnalyticalCI, MetricWithAnalyticalVar):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name="AUROC",
            reference_class="self",  # ?????? https://proceedings.neurips.cc/paper_files/paper/2019/file/73e0f7487b8e5297182c5a711d20bf26-Paper.pdf
            needs_pos_and_neg=True,
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
        y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate)

        if (not return_ci) and (not return_var):
            return auroc(y_true.to_numpy(), y_pred_prob.to_numpy())
        
        elif (not return_ci) and return_var:
            val, var = self.get_variance(df, group_mask=mask, y_true=y_true, y_pred_prob=y_pred_prob, validate=validate, return_val=True) # type: ignore
            return val, var

        elif return_ci and not return_var:
            val, ci = self.get_ci(df, group_mask=mask, y_true=y_true, y_pred_prob=y_pred_prob, validate=validate, return_val=True)
            return val, ci # type: ignore
        
        else:
            raise NotImplementedError

    
    def plot_supporting_curve(
            self, 
            metric_results_df: pd.DataFrame, 
            test_df: pd.DataFrame, 
            group_colors_dict: Optional[dict[str, str]] = None, 
            add_all_group: bool = True,
            threshold: Optional[float] = None,
            ) -> tuple[go.Figure, list[BaseTraceType], dict]:
        
        plot_groups_roc = select_extreme_groups(metric_results_df, 
                                                cols={self.metric_name_low: False, self.metric_name_high: True},
                                                n_groups=3, add_all_group=add_all_group, sort=True)
        fig, roc_traces, layout_kwargs = roc_diag(test_df, plot_groups=plot_groups_roc, group_color_dict=group_colors_dict, legend=False, threshold=threshold)
        return fig, roc_traces, layout_kwargs
    
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
        return_val: Optional[bool] = False,
        method: str = "auto"
        ) -> tuple[float, float] | tuple[float, tuple[float, float]]:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        if y_pred_prob is None:
            y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate)

        if method.upper() == "AUTO":
            if mask.sum() <= 50 or y_pred_prob[y_true.astype(bool)].min() > y_pred_prob[~y_true.astype(bool)].max():
                # Small sample size = DeLong really bad
                # Perfect separation = DeLong really bad
                # https://doi.org/10.1177/0962280215602040
                method = "newcombe"
            else:
                # I am actually not entirely sure whether delong is ever *better* than newcombe?
                # It might yield tighter CIs, since newcombe does not take the actual score distribution into account?
                # I have not seen a paper showing this, though.
                method = "delong"

        if method.upper() == "NEWCOMBE":
            auc = auroc(y_true.to_numpy(), y_pred_prob.to_numpy())
            ci = newcombe_auroc_ci(auc, y_true.to_numpy(), ci_alpha=1-ci_alpha)

        elif method.upper() == "DELONG":

            if y_true.sum() <= 1 or y_true.sum() >= len(y_true)-1:
                auc, ci = np.nan, (np.nan, np.nan)

            else:
                auc, variance = self.get_variance(df, group_mask=mask, validate=validate, y_true=y_true, y_pred_prob=y_pred_prob, return_val=True) # type: ignore
                alpha = 1 - ci_alpha
                z = scipy.stats.norm.ppf(1 - alpha / 2)

                # This assumes a normal distribution.
                # That's a relatively standard approach, also implemented e.g. in the pROC R package.
                # Apparently, this assumption is also not completely unfounded:
                # https://stats.stackexchange.com/a/361647/131402
                # It does however sometimes result in CIs outside [0, 1], which is why we need the clipping below.
                ci = max(auc - z * np.sqrt(variance), 0), min(auc + z * np.sqrt(variance), 1)

                assert 0 <= auc <= 1
                assert 0 <= ci[0] <= 1
                assert 0 <= ci[1] <= 1
                assert ci[0] <= ci[1]

        else:
            raise NotImplementedError("Valid methods are 'auto', 'newcombe', 'delong'.")

        if return_val:
            return auc, ci
        else:
            return ci 

    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
        y_pred_prob: Optional[pd.Series] = None,
        return_val: Optional[bool] = False,
        method: str = "auto",
        ) -> float | tuple[float, float]:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        if y_pred_prob is None:
            y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate)

        if method.upper() == "AUTO":
            if mask.sum() <= 50 or y_pred_prob[y_true.astype(bool)].min() > y_pred_prob[~y_true.astype(bool)].max():
                # Small sample size = DeLong really bad
                # Perfect separation = DeLong really bad
                # https://doi.org/10.1177/0962280215602040
                method = "newcombe"
            else:
                # I am actually not entirely sure whether delong is ever *better* than newcombe?
                # It might yield tighter CIs, since newcombe does not take the actual score distribution into account?
                # I have not seen a paper showing this, though.
                method = "delong"

        if method.upper() == "NEWCOMBE":
            ci_alpha = 0.95
            val, ci = self.get_ci(df, group_mask=mask, validate=validate, y_true=y_true, y_pred_prob=y_pred_prob, return_val=True, method="newcombe", ci_alpha=ci_alpha) # type: ignore

            # Estimate variance using normal approximation
            # CI width = 2 * z_α/2 * SE, so SE = CI_width / (2 * z_α/2)
            # variance = SE²
            z_alpha_2 = scipy.stats.norm.ppf((1 - ci_alpha)/2)
            var = ((ci[1] - ci[0]) / (2 * z_alpha_2))**2

        if y_true.sum() <= 1 or y_true.sum() >= len(y_true)-1:
            # Technically, this is only true in the case where there are 0 positives or negatives.
            # However, the DeLong algorithm used below also fails for the npos=1 or nneg=1 cases.
            # Since the variance will be huge in this case, anyway, it doesn't seem too bad to just reject this case, as well.
            val, var = np.nan, np.nan

        else:
            # Caution: DeLong (unrealistically) returns var=0 if auroc=1.
            val, var = delong_roc_variance(y_true.astype(int).values, y_pred_prob.values)
            assert 0 <= val <= 1.001
            assert var >= 0
            val = min(val, 1.0)  # https://github.com/jacobgil/confidenceinterval/issues/15

        if return_val:
            return val, var
        else:
            return var