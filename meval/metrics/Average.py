import numpy as np
import pandas as pd

from ..group_filter import GroupFilter
from .ComparisonMetric import ComparisonMetric, DecompositionTerm, MaskLike, MetricWithAnalyticalVar


class Average(MetricWithAnalyticalVar):

    def __init__(self, metric_col, test: bool = False, direction: str = "N/A"):
        # Average() wraps an arbitrary caller-provided column (e.g. a
        # per-sample margin, error, or custom score), so unlike most other
        # metrics it can't infer a sensible "maximize"/"minimize" direction
        # from its own definition - callers who know what metric_col means
        # can pass direction explicitly, e.g. Average("margin", direction="maximize").
        super().__init__(
            req_cols=[metric_col],
            metric_name="Avg(" + metric_col + ")",
            reference_class="self",
            needs_all_classes=False,
            is_descriptive=False,
            direction=direction,
            test=test
        )
        self.metric_col = metric_col

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None, 
        group_mask: MaskLike | None = None,
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

    def decompose(self, df: pd.DataFrame, validate: bool = True) -> list[DecompositionTerm] | None:
        mask = self.get_group_mask(df, validate=validate)
        vals = np.asarray(ComparisonMetric._masked_array(df[self.metric_col], mask), dtype=float)
        return [DecompositionTerm(1.0, vals, np.isfinite(vals))]


    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        validate: bool = True,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred: pd.Series | np.ndarray | None = None,
        y_pred_prob: pd.Series | np.ndarray | None = None,
        return_val: bool | None = False
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