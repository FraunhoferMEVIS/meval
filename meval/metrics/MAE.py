import warnings

import numpy as np
import pandas as pd

from ..group_filter import GroupFilter
from ..stats.utils import mean_and_variance_of_mean
from .ComparisonMetric import ComparisonMetric, DecompositionTerm, MaskLike, MetricWithAnalyticalVar


class MAE(MetricWithAnalyticalVar):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_float_pred_cols],
            metric_name='MAE',
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            direction="minimize",
            test=test
        )

    @property
    def description(self) -> str | None:
        return "Mean absolute difference between prediction and target over the group's cases. Lower is better."

    @staticmethod
    def _looks_like_binary_classification_inputs(
        y_true_np: np.ndarray,
        y_pred_np: np.ndarray,
    ) -> bool:
        finite_y_true = y_true_np[np.isfinite(y_true_np)]
        finite_y_pred = y_pred_np[np.isfinite(y_pred_np)]

        if finite_y_true.size == 0 or finite_y_pred.size == 0:
            return False

        unique_y_true = np.unique(finite_y_true)
        return bool(
            np.all(np.isin(unique_y_true, [0.0, 1.0]))
            and np.all((finite_y_pred >= 0.0) & (finite_y_pred <= 1.0))
        )

    def check_suspicious_usage(self, df: pd.DataFrame) -> None:
        try:
            y_true_np = np.asarray(self.get_float_y_true(df, validate=False, return_array=True), dtype=float)
            y_pred_np = np.asarray(self.get_float_y_pred(df, validate=False, return_array=True), dtype=float)
        except Exception:
            return

        if self._looks_like_binary_classification_inputs(y_true_np, y_pred_np):
            warnings.warn(
                "MAE was requested on data with binary 0/1 y_true and y_pred values in [0, 1]. "
                "meval allows this because MAE treats inputs as numeric regression targets, "
                "but for binary classification Accuracy or BrierScore is usually more appropriate; "
                "AUROC is only appropriate when you have continuous score/probability predictions.",
                UserWarning,
                stacklevel=3,
            )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None, 
        group_mask: MaskLike | None = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | tuple[float, float]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true_np = self.get_float_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred_np = self.get_float_y_pred(df, mask=mask, validate=validate, return_array=True)
        errs = np.abs(y_true_np - y_pred_np)
        mae, var = mean_and_variance_of_mean(errs)

        if return_var:
            return mae, var
        else:
            return mae

    def decompose(self, df: pd.DataFrame, validate: bool = True) -> list[DecompositionTerm] | None:
        mask = self.get_group_mask(df, validate=validate)
        y_true_np = self.get_float_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred_np = self.get_float_y_pred(df, mask=mask, validate=validate, return_array=True)
        errs = np.abs(np.asarray(y_true_np, dtype=float) - np.asarray(y_pred_np, dtype=float))
        return [DecompositionTerm(1.0, errs, np.isfinite(errs))]

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

        assert y_pred_prob is None, "MAE does not use y_pred_prob, expected y_pred_prob to be None."

        mask: MaskLike | None
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
        val, var = mean_and_variance_of_mean(errs)

        if return_val:
            return val, var
        else:
            return var

