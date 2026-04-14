from typing import Optional
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go
import pandas as pd
from confidenceinterval.delong import delong_roc_variance
import numpy as np
import scipy.stats

from .ComparisonMetric import ComparisonMetric, CurveBasedComparisonMetric, MetricWithAnalyticalCI, MetricWithAnalyticalVar, MaskLike
from .fastauc import fast_numba_auc
from ..diags import roc_diag
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups
from ..config import settings
from ..stats import newcombe_auroc_ci


def _fast_binary_auroc_with_min_cases(
    y_true: pd.Series | np.ndarray,
    y_pred_prob: pd.Series | np.ndarray,
    min_cases_per_class: int = 3,
) -> float:
    y_true_np = np.asarray(y_true)
    y_pred_prob_np = np.asarray(y_pred_prob, dtype=float)

    n_pos = y_true_np.sum()
    n_neg = len(y_true_np) - n_pos
    if n_pos < min_cases_per_class or n_neg < min_cases_per_class:
        return np.nan

    return fast_numba_auc(y_true_np, y_pred_prob_np)


def _to_binary_arrays(
    y_true: pd.Series | np.ndarray,
    y_pred_prob: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_np = np.asarray(y_true, dtype=bool)
    y_pred_prob_np = np.asarray(y_pred_prob, dtype=float)
    assert y_true_np.ndim == 1
    assert y_pred_prob_np.ndim == 1
    assert y_true_np.shape[0] == y_pred_prob_np.shape[0]
    return y_true_np, y_pred_prob_np


def _choose_auto_method(y_true_np: np.ndarray, y_pred_prob_np: np.ndarray) -> str:
    n_samples = y_true_np.shape[0]

    # Small sample size = DeLong really bad
    # https://doi.org/10.1177/0962280215602040
    if n_samples <= 50:
        return "newcombe"

    pos_scores = y_pred_prob_np[y_true_np]
    neg_scores = y_pred_prob_np[~y_true_np]

    # Perfect separation = DeLong really bad
    # https://doi.org/10.1177/0962280215602040
    if pos_scores.min() > neg_scores.max():
        return "newcombe"

    # I am actually not entirely sure whether delong is ever *better* than newcombe?
    # It might/should? yield tighter CIs, since newcombe does not take the actual score distribution into account?
    # I have not yet seen a paper actually demonstrating this, though.
    return "delong"


class AUROC(CurveBasedComparisonMetric, MetricWithAnalyticalCI, MetricWithAnalyticalVar):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name="AUROC",
            reference_class="self",  # ?????? https://proceedings.neurips.cc/paper_files/paper/2019/file/73e0f7487b8e5297182c5a711d20bf26-Paper.pdf
            needs_all_classes=True,
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
        y_true_np = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred_prob_np = self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True)

        if (not return_ci) and (not return_var):
            return _fast_binary_auroc_with_min_cases(y_true_np, y_pred_prob_np, min_cases_per_class=settings.auroc_min_cases_per_class)

        if (not return_ci) and return_var:
            variance_out = self.get_variance(df, group_mask=mask, y_true=y_true_np, y_pred_prob=y_pred_prob_np, validate=validate, return_val=True)
            assert isinstance(variance_out, tuple)
            val, var = variance_out
            return val, var

        if return_ci and not return_var:
            ci_out = self.get_ci(df, group_mask=mask, y_true=y_true_np, y_pred_prob=y_pred_prob_np, validate=validate, return_val=True)
            assert isinstance(ci_out, tuple)
            val, ci = ci_out
            assert isinstance(ci, tuple)
            return val, ci
        
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
        group_mask: Optional[MaskLike] = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        y_true: Optional[pd.Series | np.ndarray] = None,
        y_pred: Optional[pd.Series | np.ndarray] = None,
        y_pred_prob: Optional[pd.Series | np.ndarray] = None,
        return_val: Optional[bool] = False,
        method: str = "auto"
        ) -> tuple[float, float] | tuple[float, tuple[float, float]]:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        min_cases_per_class = settings.auroc_min_cases_per_class
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        if y_pred_prob is None:
            y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True)

        y_true_np, y_pred_prob_np = _to_binary_arrays(y_true, y_pred_prob)

        if method.upper() == "AUTO":
            method = _choose_auto_method(y_true_np, y_pred_prob_np)

        if method.upper() == "NEWCOMBE":
            auc = _fast_binary_auroc_with_min_cases(y_true_np, y_pred_prob_np, min_cases_per_class=min_cases_per_class)
            ci_list = newcombe_auroc_ci(auc, y_true_np, ci_alpha=1-ci_alpha)
            ci = (float(ci_list[0]), float(ci_list[1]))

        elif method.upper() == "DELONG":

            n_pos = int(y_true_np.sum())
            if n_pos < min_cases_per_class or n_pos > len(y_true_np) - min_cases_per_class:
                auc, ci = np.nan, (np.nan, np.nan)

            else:
                variance_out = self.get_variance(df, group_mask=mask, validate=validate, y_true=y_true_np, y_pred_prob=y_pred_prob_np, return_val=True, method="delong")
                assert isinstance(variance_out, tuple)
                auc, variance = variance_out
                alpha = 1 - ci_alpha
                z = scipy.stats.norm.ppf(1 - alpha / 2)
                auc = float(auc)
                variance = float(variance)

                # This assumes a normal distribution.
                # That's a relatively standard approach, also implemented e.g. in the pROC R package.
                # Apparently, this assumption is also not completely unfounded:
                # https://stats.stackexchange.com/a/361647/131402
                # It does however sometimes result in CIs outside [0, 1], which is why we need the clipping below.
                ci = (float(max(auc - z * np.sqrt(variance), 0)), float(min(auc + z * np.sqrt(variance), 1)))

                assert 0 <= auc <= 1
                assert 0 <= ci[0] <= 1
                assert 0 <= ci[1] <= 1
                assert ci[0] <= ci[1]

        else:
            raise NotImplementedError("Valid methods are 'auto', 'newcombe', 'delong'.")

        if return_val:
            return float(auc), ci
        else:
            return ci 

    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        y_true: Optional[pd.Series | np.ndarray] = None,
        y_pred: Optional[pd.Series | np.ndarray] = None,
        y_pred_prob: Optional[pd.Series | np.ndarray] = None,
        return_val: Optional[bool] = False,
        method: str = "auto",
        ) -> float | tuple[float, float]:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        min_cases_per_class = settings.auroc_min_cases_per_class
        if y_true is None:
            y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        if y_pred_prob is None:
            y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True)

        y_true_np, y_pred_prob_np = _to_binary_arrays(y_true, y_pred_prob)

        if method.upper() == "AUTO":
            method = _choose_auto_method(y_true_np, y_pred_prob_np)

        if method.upper() == "NEWCOMBE":
            ci_alpha = 0.95
            val, ci = self.get_ci(df, group_mask=mask, validate=validate, y_true=y_true_np, y_pred_prob=y_pred_prob_np, return_val=True, method="newcombe", ci_alpha=ci_alpha)
            val = float(val)

            # Estimate variance using normal approximation
            # CI width = 2 * z_α/2 * SE, so SE = CI_width / (2 * z_α/2)
            # variance = SE²
            z_alpha_2 = scipy.stats.norm.ppf((1 - ci_alpha)/2)
            assert isinstance(ci, tuple)
            var = float(((ci[1] - ci[0]) / (2 * z_alpha_2))**2)

        elif method.upper() == "DELONG":
            n_pos = int(y_true_np.sum())
            if n_pos < min_cases_per_class or n_pos > len(y_true_np) - min_cases_per_class:
                # Technically, this is only true in the case where there are 0 positives or negatives.
                # However, the DeLong algorithm used below also fails for the npos=1 or nneg=1 cases.
                # Since the variance will be huge in this case, anyway, it doesn't seem too bad to just reject this case, as well.
                val, var = np.nan, np.nan

            else:
                # Caution: DeLong (unrealistically) returns var=0 if auroc=1.
                val, var = delong_roc_variance(y_true_np.astype(int), y_pred_prob_np)
                assert 0 <= val <= 1.001
                assert var >= 0
                val = float(min(val, 1.0))  # https://github.com/jacobgil/confidenceinterval/issues/15
                var = float(var)

        else:
            raise NotImplementedError("Valid methods are 'auto', 'newcombe', 'delong'.")

        if return_val:
            return val, var
        else:
            return var


