import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from confidenceinterval.delong import delong_roc_variance
from plotly.basedatatypes import BaseTraceType

from ..config import settings
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups
from ..stats.confidence_intervals import newcombe_auroc_ci
from .ComparisonMetric import (
    ComparisonMetric,
    CurveBasedComparisonMetric,
    MaskLike,
    MetricWithAnalyticalCI,
    MetricWithAnalyticalVar,
)
from .fastauc import fast_numba_auc


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


def _has_only_hard_binary_predictions(y_pred_prob: pd.Series | np.ndarray) -> bool:
    y_pred_prob_np = np.asarray(y_pred_prob, dtype=float)
    finite_values = y_pred_prob_np[np.isfinite(y_pred_prob_np)]

    if finite_values.size == 0:
        return False

    unique_values = np.unique(finite_values)
    return bool(np.all(np.isin(unique_values, [0.0, 1.0])))


def _raise_if_hard_binary_predictions(y_pred_prob: pd.Series | np.ndarray) -> None:
    if _has_only_hard_binary_predictions(y_pred_prob):
        raise ValueError(
            "AUROC requires continuous score/probability predictions, but the provided "
            "y_pred_prob values are purely 0/1. This looks like hard class predictions. "
            "Use Accuracy or Specificity/Recall or provide score/probability "
            "predictions for AUROC."
        )


def _choose_auto_method(y_true_np: np.ndarray, y_pred_prob_np: np.ndarray) -> str:
    n_samples = y_true_np.shape[0]

    # Small sample size = DeLong really bad
    # https://doi.org/10.1177/0962280215602040
    if n_samples <= 50:
        return "newcombe"

    pos_scores = y_pred_prob_np[y_true_np]
    neg_scores = y_pred_prob_np[~y_true_np]

    if pos_scores.size == 0 or neg_scores.size == 0:
        # does not really matter; both methods should hardcode nan for this case.
        # We need to explicitly catch this though because the next if clause will fail in this case.
        return "newcombe"

    # Perfect separation = DeLong really bad
    # https://doi.org/10.1177/0962280215602040
    if pos_scores.min() > neg_scores.max():
        return "newcombe"

    # I am actually not entirely sure whether delong is ever *better* than newcombe?
    # It might/should? yield tighter CIs, since newcombe does not take the actual score distribution into account?
    # I have not yet seen a paper actually demonstrating this, though.
    return "delong"


def _delong_auroc_and_variance(y_true_np: np.ndarray, y_pred_prob_np: np.ndarray) -> tuple[float, float]:
    min_cases_per_class = settings.auroc_min_cases_per_class
    n_pos = int(y_true_np.sum())
    if n_pos < min_cases_per_class or n_pos > len(y_true_np) - min_cases_per_class:
        # Technically, this is only true in the case where there are 0 positives or negatives.
        # However, the DeLong algorithm used below also fails for the npos=1 or nneg=1 cases.
        # Since the variance will be huge in this case, anyway, it doesn't seem too bad to just reject this case, as well.
        return np.nan, np.nan

    # Caution: DeLong (unrealistically) returns var=0 if auroc=1.
    val, var = delong_roc_variance(y_true_np.astype(int), y_pred_prob_np)
    assert 0 <= val <= 1.001
    assert var >= 0
    return float(min(val, 1.0)), float(var)  # https://github.com/jacobgil/confidenceinterval/issues/15


def _newcombe_auroc_and_ci(
    y_true_np: np.ndarray, y_pred_prob_np: np.ndarray, ci_alpha: float
) -> tuple[float, tuple[float, float]]:
    auc = _fast_binary_auroc_with_min_cases(y_true_np, y_pred_prob_np, min_cases_per_class=settings.auroc_min_cases_per_class)
    ci_list = newcombe_auroc_ci(auc, y_true_np, ci_alpha=1-ci_alpha)
    return auc, (float(ci_list[0]), float(ci_list[1]))


def _reject_complement_mask(metric: ComparisonMetric, complement_mask: MaskLike | None) -> None:
    if complement_mask is not None:
        raise NotImplementedError(f"{type(metric).__name__} does not depend on the group's complement; complement_mask cannot be used.")


class _AUROCBase(MetricWithAnalyticalCI, MetricWithAnalyticalVar):
    """Value, CI and variance of an AUROC. Subclasses can change which cases it is computed over
    by overriding _cases()."""

    def _cases(
        self,
        df: pd.DataFrame,
        group_filter: GroupFilter | None,
        group_mask: MaskLike | None,
        validate: bool,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred_prob: pd.Series | np.ndarray | None = None,
        complement_mask: MaskLike | None = None,
    ) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
        """y_true and y_pred_prob of the cases the AUROC is computed over: by default, the group's.

        Arrays passed by the caller replace what would be read from df. A subclass that selects
        other cases must override this and decide what such arrays mean for it. complement_mask is
        for subclasses whose value depends on the group's complement."""
        _reject_complement_mask(self, complement_mask)
        if y_true is None or y_pred_prob is None:
            mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
            if y_true is None:
                y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
            if y_pred_prob is None:
                y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True)
        return y_true, y_pred_prob

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None, 
        group_mask: MaskLike | None = None,
        validate: bool = True,
        return_ci: bool = False,
        return_var: bool = False,
        complement_mask: MaskLike | None = None,
        ) -> float | tuple[float, float] | tuple[float, tuple[float, float]]:

        if return_ci and return_var:
            raise NotImplementedError

        if return_ci:
            return self.get_ci(df, group_filter, group_mask, validate=validate, return_val=True, complement_mask=complement_mask)  # type: ignore[return-value]

        if return_var:
            return self.get_variance(df, group_filter, group_mask, validate=validate, return_val=True, complement_mask=complement_mask)  # type: ignore[return-value]

        y_true_np, y_pred_prob_np = self._cases(df, group_filter, group_mask, validate, complement_mask=complement_mask)
        return _fast_binary_auroc_with_min_cases(y_true_np, y_pred_prob_np, min_cases_per_class=settings.auroc_min_cases_per_class)

    def get_ci(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred: pd.Series | np.ndarray | None = None,
        y_pred_prob: pd.Series | np.ndarray | None = None,
        return_val: bool | None = False,
        method: str = "auto",
        complement_mask: MaskLike | None = None,
        ) -> tuple[float, float] | tuple[float, tuple[float, float]]:

        y_true_np, y_pred_prob_np = _to_binary_arrays(*self._cases(df, group_filter, group_mask, validate, y_true, y_pred_prob, complement_mask))

        if method.upper() == "AUTO":
            method = _choose_auto_method(y_true_np, y_pred_prob_np)

        if method.upper() == "NEWCOMBE":
            auc, ci = _newcombe_auroc_and_ci(y_true_np, y_pred_prob_np, ci_alpha)

        elif method.upper() == "DELONG":
            auc, variance = _delong_auroc_and_variance(y_true_np, y_pred_prob_np)

            if np.isnan(auc):
                ci = (np.nan, np.nan)

            else:
                alpha = 1 - ci_alpha
                z = scipy.stats.norm.ppf(1 - alpha / 2)

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
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        validate: bool = True,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred: pd.Series | np.ndarray | None = None,
        y_pred_prob: pd.Series | np.ndarray | None = None,
        return_val: bool | None = False,
        method: str = "auto",
        complement_mask: MaskLike | None = None,
        ) -> float | tuple[float, float]:

        y_true_np, y_pred_prob_np = _to_binary_arrays(*self._cases(df, group_filter, group_mask, validate, y_true, y_pred_prob, complement_mask))

        if method.upper() == "AUTO":
            method = _choose_auto_method(y_true_np, y_pred_prob_np)

        if method.upper() == "NEWCOMBE":
            ci_alpha = 0.95
            val, ci = _newcombe_auroc_and_ci(y_true_np, y_pred_prob_np, ci_alpha)
            val = float(val)

            # Estimate variance using normal approximation
            # CI width = 2 * z_α/2 * SE, so SE = CI_width / (2 * z_α/2)
            # variance = SE²
            z_alpha_2 = scipy.stats.norm.ppf((1 - ci_alpha)/2)
            var = float(((ci[1] - ci[0]) / (2 * z_alpha_2))**2)

        elif method.upper() == "DELONG":
            val, var = _delong_auroc_and_variance(y_true_np, y_pred_prob_np)

        else:
            raise NotImplementedError("Valid methods are 'auto', 'newcombe', 'delong'.")

        if return_val:
            return val, var
        else:
            return var


class AUROC(CurveBasedComparisonMetric, _AUROCBase):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name="AUROC",
            reference_class="self",  # ?????? https://proceedings.neurips.cc/paper_files/paper/2019/file/73e0f7487b8e5297182c5a711d20bf26-Paper.pdf
            needs_all_classes=True,
            is_descriptive=False,
            direction="maximize",
            test=test
        )

    @property
    def description(self) -> str | None:
        return ("Probability that a random positive of the group scores higher than a random negative of the same "
                "group. Compares cases within the group only.")

    def check_suspicious_usage(self, df: pd.DataFrame) -> None:
        y_pred_prob_np = self.get_binary_y_pred_prob(df, validate=False, return_array=True)
        _raise_if_hard_binary_predictions(y_pred_prob_np)

    def plot_supporting_curve(
            self, 
            metric_results_df: pd.DataFrame, 
            test_df: pd.DataFrame, 
            group_colors_dict: dict[str, str] | None = None, 
            add_all_group: bool = True,
            threshold: float | None = None,
            ) -> tuple[go.Figure, list[BaseTraceType], dict]:
        
        plot_groups_roc = select_extreme_groups(metric_results_df, 
                                                cols={self.metric_name_low: False, self.metric_name_high: True},
                                                n_groups=3, add_all_group=add_all_group, sort=True)
        from ..diags import roc_diag

        fig, roc_traces, layout_kwargs = roc_diag(test_df, plot_groups=plot_groups_roc, group_color_dict=group_colors_dict, legend=False, threshold=threshold)
        return fig, roc_traces, layout_kwargs
