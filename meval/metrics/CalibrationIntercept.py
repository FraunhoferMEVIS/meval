import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from ..config import settings
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups
from ..stats.cache import dataset_computation, shared_computation
from ._calibration import (
    calibration_bias_at_reference,
    fit_calibration_slope_intercept,
    mean_reference_logit,
    wald_ci,
)
from .ComparisonMetric import (
    ComparisonMetric,
    CurveBasedComparisonMetric,
    MaskLike,
    MetricWithAnalyticalCI,
    MetricWithAnalyticalVar,
)


class CalibrationIntercept(CurveBasedComparisonMetric, MetricWithAnalyticalCI, MetricWithAnalyticalVar):
    """
    Signed calibration bias in log-odds: how much the model systematically
    over- or under-states risk for a group, evaluated at a reference score
    level shared by all groups. Positive means the model systematically
    understates risk for this group (true log-odds run higher than predicted),
    negative means it overstates risk.

    Computed from the per-group recalibration fit logit(y) ~ a + b*logit(p) as

        Delta(L0) = a + (b - 1) * L0

    where L0 is the reference point (see `reference_score` below). This is the
    gap between the fitted truth and the model's own claim at score level L0.

    Why a reference point is needed
    -------------------------------
    The plain intercept `a` of the joint fit is not a distribution-free summary
    of "level bias": it is by construction the fitted log-odds at logit(p) = 0,
    i.e. exactly at p = 0.5. For a group whose predicted risks cluster far from
    0.5 - which is the norm for low-prevalence subgroups - that is an
    extrapolation to a score level the group has little or no data at, making
    `a` both high-variance and highly sensitive to small errors in the slope.

    The natural alternative, calibration-in-the-large (refitting with the slope
    constrained to 1, cf. Steyerberg & Vergouwe 2014), removes that problem but
    introduces a subtler one: when the true slope is not 1, the constrained fit
    converges to a projection whose implicit evaluation point is determined by
    *that group's own score distribution*. Two groups with identical true
    calibration but different score distributions then get different values, so
    the cross-group comparison is confounded exactly where it matters.

    Evaluating Delta at an explicitly shared L0 fixes both issues: the estimand
    depends only on (a, b) and L0, never on the group's score distribution, so
    it is directly comparable across groups with differing prevalences and
    score spreads. The cost is variance rather than bias - the free-slope fit
    estimates two parameters instead of one, and the L0^2 * Var(b) term grows
    if L0 sits far from a group's own scores - but that penalty shrinks with n,
    whereas the constrained fit's bias does not.

    Interpreting alongside CalibrationSlope
    ---------------------------------------
    When a group's slope differs markedly from its comparator's, their two
    calibration lines cross, and no single reference point can summarise the
    difference: the sign of the bias can flip across the score range. Delta(L0)
    remains well defined and comparable, but should then be read together with
    CalibrationSlope, and ideally at more than one reference point, rather than
    as a complete description of the disagreement.

    See meval/metrics/_calibration.py for the fitting logic and full
    references. The underlying fit is shared with CalibrationSlope via
    shared_computation, so requesting both does not refit the model twice.
    """

    def __init__(self, test: bool = False, reference_score: str | float = "mean"):
        """
        :param test: whether to run significance testing for this metric.
        :param reference_score: the score level at which calibration bias is
            evaluated, as a probability.
            - "mean" (default): use the mean predicted logit over the whole
              evaluation dataset. Shared by all groups and all permutations,
              and sits where the data actually is, which keeps the variance
              of the estimate low.
            - a float in (0, 1): use this fixed probability instead.
              `reference_score=0.5` reproduces the plain joint-fit intercept
              `a` (since logit(0.5) = 0).
        """
        if not isinstance(reference_score, str):
            if not 0.0 < float(reference_score) < 1.0:
                raise ValueError(
                    f"reference_score must be 'mean' or a probability strictly between 0 and 1, got {reference_score}"
                )
        elif reference_score != "mean":
            raise ValueError(f"reference_score must be 'mean' or a float in (0, 1), got {reference_score!r}")

        self.reference_score = reference_score

        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name='Calibration intercept',
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            direction="N/A",
            test=test,
            curve_family="reliability",
        )

    @property
    def description(self) -> str | None:
        at = ("the dataset's mean predicted log-odds" if self.reference_score == "mean"
              else f"a predicted risk of {float(self.reference_score):.2g}")
        return (f"Systematic calibration bias for the group in log-odds, at {at}: above 0 = the model understates the "
                "group's risk, below 0 = it overstates it.")

    def _get_reference_logit(self, df: pd.DataFrame, validate: bool) -> float:
        """
        The reference point L0 on the logit scale. For reference_score="mean"
        this is a property of the whole dataset (not of any group or mask), so
        it is computed once per compare_groups() run and cached dataset-wide;
        this also keeps it fixed across permutations during significance
        testing, as it must be.

        :param df: the full dataframe passed to the metric call.
        :param validate: whether to validate df against the expected schema.
        :return: L0, the reference score level on the logit scale.
        """
        if self.reference_score != "mean":
            return float(np.log(self.reference_score / (1.0 - self.reference_score)))

        return dataset_computation(
            "calibration_mean_reference_logit",
            df,
            lambda: mean_reference_logit(
                self.get_binary_y_pred_prob(df, mask=None, validate=validate, return_array=True)
            ),
        )

    def _get_fit(self, df: pd.DataFrame, group_filter: GroupFilter | None, mask: MaskLike, validate: bool):
        """
        Fetch this group's shared CalibrationFit (see _calibration.py), computing
        it on first access and reusing it on subsequent calls and across
        sibling metrics (e.g. CalibrationSlope) for the same group.

        :param df: the full dataframe passed to the metric call.
        :param group_filter: identifies the group to fit on; used by
            shared_computation to key the cache (see meval/stats/cache.py).
        :param mask: boolean mask selecting this group's rows in df.
        :param validate: whether to validate df against the expected schema.
        :return: a CalibrationFit, or None if the fit is undefined/unreliable.
        """
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True)
        return shared_computation(
            "calibration_slope_intercept_fit",
            df,
            group_filter,
            lambda: fit_calibration_slope_intercept(
                y_true, y_pred_prob, min_cases_per_class=settings.calibration_min_cases_per_class
            ),
        )

    def _get_bias(self, df: pd.DataFrame, group_filter: GroupFilter | None, mask: MaskLike, validate: bool):
        """
        This group's calibration bias at the shared reference point, and its
        variance, or (nan, nan) if the underlying fit is undefined/unreliable.

        :param df: the full dataframe passed to the metric call.
        :param group_filter: identifies the group to evaluate on.
        :param mask: boolean mask selecting this group's rows in df.
        :param validate: whether to validate df against the expected schema.
        :return: (Delta(L0), Var(Delta(L0))).
        """
        fit = self._get_fit(df, group_filter, mask, validate)
        if fit is None:
            return np.nan, np.nan
        return calibration_bias_at_reference(fit, self._get_reference_logit(df, validate))

    def __call__(
        self,
        df: pd.DataFrame,
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        validate: bool = True
        ) -> float:
        """
        :param df: dataframe containing y_true and y_pred_prob columns.
        :param group_filter: identifies the group to evaluate on.
        :param group_mask: alternative to group_filter; a boolean mask
            selecting this group's rows. NOTE: shared_computation uses
            group_filter to cache the fit, so calls with only group_mask still
            work but do not share the fit with sibling metrics.
        :param validate: whether to validate df against the expected schema.
        :return: the calibration bias at the reference point, or NaN if
            undefined/unreliable.
        """
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        val, _ = self._get_bias(df, group_filter, mask, validate)
        return val

    def get_ci(
        self,
        df: pd.DataFrame,
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        return_val: bool | None = False,
        ) -> tuple[float, float] | tuple[float, tuple[float, float]]:
        """
        Analytical Wald confidence interval for the calibration bias, derived
        from the recalibration fit's covariance matrix via the linear contrast
        [1, L0] (see calibration_bias_at_reference).

        :param df: dataframe containing y_true and y_pred_prob columns.
        :param group_filter: identifies the group to evaluate on.
        :param group_mask: alternative to group_filter; see __call__.
        :param ci_alpha: confidence level, e.g. 0.95 for a 95% CI.
        :param validate: whether to validate df against the expected schema.
        :param return_val: if True, also return the point estimate.
        :return: (low, high) CI bounds, or (value, (low, high)) if return_val.
        """
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        val, var = self._get_bias(df, group_filter, mask, validate)

        ci = (np.nan, np.nan) if np.isnan(var) else wald_ci(val, var, ci_alpha=ci_alpha)

        return (val, ci) if return_val else ci

    def get_variance(
        self,
        df: pd.DataFrame,
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        validate: bool = True,
        return_val: bool | None = False,
        ) -> float | tuple[float, float]:
        """
        Analytical variance of the calibration bias at the reference point,
        Var(a) + L0^2 Var(b) + 2 L0 Cov(a, b), from the fit's covariance matrix.

        :param df: dataframe containing y_true and y_pred_prob columns.
        :param group_filter: identifies the group to evaluate on.
        :param group_mask: alternative to group_filter; see __call__.
        :param validate: whether to validate df against the expected schema.
        :param return_val: if True, also return the point estimate.
        :return: the variance, or (value, variance) if return_val.
        """
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        val, var = self._get_bias(df, group_filter, mask, validate)

        return (val, var) if return_val else var

    def plot_supporting_curve(
            self,
            metric_results_df: pd.DataFrame,
            test_df: pd.DataFrame,
            group_colors_dict: dict[str, str] | None = None,
            add_all_group: bool = True,
            threshold: float | None = None,
            ) -> tuple[go.Figure, list[BaseTraceType], dict]:
        """
        Reliability diagram for the groups with the most extreme calibration
        bias, shared with DRMSCE/CalibrationSlope's supporting curve
        (curve_family="reliability"): only whichever of these metrics is
        listed first in a compare_groups() metrics=[...] call actually
        produces this plot (see plot_metric_overview's curve_family dedup);
        the others are skipped to avoid duplicate reliability diagrams.

        :param metric_results_df: per-group metric results, used to select
            which groups' curves to plot.
        :param test_df: the full dataframe the curves are computed from.
        :param group_colors_dict: optional mapping from group name to plot color.
        :param add_all_group: whether to always include the "all" group's curve.
        :param threshold: optional decision threshold to mark on the plot.
        :return: (figure, traces, layout_kwargs) as expected by plot_metric_overview.
        """
        plot_groups_rel = select_extreme_groups(metric_results_df,
                                                cols={self.metric_name_low: False, self.metric_name_high: True},
                                                n_groups=3, add_all_group=add_all_group, sort=True)
        from ..diags import rel_diag

        fig, rel_traces, layout_kwargs = rel_diag(test_df, plot_groups_rel, group_color_dict=group_colors_dict, add_risk_density=False,
                                                  legend=False, fig_title=None, threshold=threshold)
        return fig, rel_traces, layout_kwargs
