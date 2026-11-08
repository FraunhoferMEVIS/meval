import numpy as np
import pandas as pd

from ..group_filter import GroupFilter
from .ComparisonMetric import ComparisonMetric, DecompositionTerm, MaskLike, MetricWithAnalyticalVar


class Average(MetricWithAnalyticalVar):
    """
    Mean of an arbitrary column within a group.

    With y_class set, the mean is taken only over the group's cases with y_true == y_class
    (binary labels: y_class 0/1 or False/True; multiclass labels: the class id).
    Average("y_pred_prob", y_class=0) is then the mean predicted probability among the group's
    negatives and y_class=1 among its positives. These are the intercept and intercept + slope
    of an OLS regression of the score on the label within the group; equal values across groups
    correspond to "balance for the negative / positive class" (Kleinberg, Mullainathan &
    Raghavan, 2016).
    """

    def __init__(self, metric_col, test: bool = False, direction: str = "N/A", y_class: int | bool | None = None):
        # Average() wraps an arbitrary caller-provided column (e.g. a
        # per-sample margin, error, or custom score), so unlike most other
        # metrics it can't infer a sensible "maximize"/"minimize" direction
        # from its own definition - callers who know what metric_col means
        # can pass direction explicitly, e.g. Average("margin", direction="maximize").
        req_cols = [metric_col] if y_class is None else [metric_col, ComparisonMetric.y_true_cols]
        metric_name = f"Avg({metric_col})" if y_class is None else f"Avg({metric_col} | y={int(y_class)})"
        super().__init__(
            req_cols=req_cols,
            metric_name=metric_name,
            reference_class="self",
            needs_all_classes=False,
            is_descriptive=False,
            direction=direction,
            test=test
        )
        self.metric_col = metric_col
        self.y_class = y_class

    @property
    def description(self) -> str | None:
        cases = "cases" if self.y_class is None else f"cases with label {int(self.y_class)}"
        return f"Mean of {self.metric_col} over the group's {cases}."

    def _class_mask(self, df: pd.DataFrame) -> np.ndarray:
        # One entry per row of df. Comparing with the class id works for bool labels
        # (False == 0, True == 1) and integer class ids alike; missing labels belong to no class.
        y_true = ComparisonMetric.get_y_true(df, validate=False, return_array=True)
        if y_true.dtype == object:  # nullable labels holding pd.NA
            return pd.Series(y_true).eq(self.y_class).fillna(False).to_numpy(dtype=bool)
        return np.asarray(y_true == self.y_class, dtype=bool)

    def _restrict_to_class(self, df: pd.DataFrame, mask: MaskLike) -> MaskLike:
        if self.y_class is None:
            return mask
        is_class = self._class_mask(df)
        mask_np = np.asarray(mask)
        if np.issubdtype(mask_np.dtype, np.integer):  # row positions, possibly repeated (bootstrap)
            return mask_np[is_class[mask_np]]
        return mask_np.astype(bool) & is_class

    def __call__(
        self,
        df: pd.DataFrame,
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | tuple[float, float]:

        mask = self._restrict_to_class(df, self.get_group_mask(df, group_filter, group_mask, validate=validate))

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
        reference = np.isfinite(vals)
        if self.y_class is not None:
            reference &= self._class_mask(df)
        return [DecompositionTerm(1.0, vals, reference)]


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

        mask = self._restrict_to_class(df, self.get_group_mask(df, group_filter, group_mask, validate=validate))
        vals = ComparisonMetric._masked_array(df[self.metric_col], mask)

        n_valid = np.isfinite(vals).sum()
        if n_valid == 0:
            val, var = np.nan, np.nan
        else:
            val = float(np.nanmean(vals))
            var = np.nan if n_valid <= 1 else float(np.nanvar(vals, ddof=1) / n_valid)

        return (val, var) if return_val else var
