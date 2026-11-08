import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from ..config import settings
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups
from ..stats.cache import shared_computation
from ._calibration import fit_calibration_slope_intercept, wald_ci
from .ComparisonMetric import (
    ComparisonMetric,
    CurveBasedComparisonMetric,
    MaskLike,
    MetricWithAnalyticalCI,
    MetricWithAnalyticalVar,
)


class CalibrationSlope(CurveBasedComparisonMetric, MetricWithAnalyticalCI, MetricWithAnalyticalVar):
    """
    Calibration slope from a per-group logistic recalibration model
    logit(y_true) ~ intercept + slope * logit(y_pred_prob): slope=1
    indicates a well-spread risk score, slope<1 overconfident (too extreme)
    predictions, slope>1 underconfident predictions. The slope was
    originally proposed (as a measure of "spread", not under the name
    "calibration slope") by:
        Cox, D.R. (1958). Two further applications of a model for binary
        regression. Biometrika, 45(3/4), 562-565.
    The interpretation used here follows standard clinical prediction
    modeling practice, see e.g.:
        Steyerberg, E.W. and Vergouwe, Y. (2014). Towards better clinical
        prediction models: seven steps for development and an ABCD for
        validation. European Heart Journal, 35(29), 1925-1931.

    The underlying fit is shared with CalibrationIntercept via
    shared_computation, so requesting both on the same group does not refit
    the model twice. See meval/metrics/_calibration.py for the fitting logic
    and full references.

    Shares curve_family="reliability" with CalibrationIntercept/DRMSCE: its
    plot_supporting_curve is only used if this metric happens to be the first
    member of that family requested (see plot_metric_overview's curve_family
    dedup), and shows the same reliability diagram those other metrics would.
    """

    def __init__(self, test: bool = False):
        """
        :param test: whether to run significance testing for this metric.
        """
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name='Calibration slope',
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            direction="N/A",
            test=test,
            curve_family="reliability",
        )

    @property
    def description(self) -> str | None:
        return ("Slope b of the group's recalibration fit logit P(y=1) = a + b logit(score): 1 = well spread, "
                "below 1 = overconfident (too extreme), above 1 = underconfident.")

    def _get_fit(self, df: pd.DataFrame, group_filter: GroupFilter | None, mask: MaskLike, validate: bool):
        """
        Fetch this group's shared CalibrationFit (see _calibration.py), computing
        it on first access and reusing it on subsequent calls and across
        sibling metrics (e.g. CalibrationIntercept) for the same group.

        :param df: the full dataframe passed to the metric call.
        :param group_filter: identifies the group to fit on; required by
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
            selecting this group's rows. NOTE: shared_computation requires
            group_filter to cache the fit, so calls with only group_mask will
            still work but will not benefit from caching with sibling metrics.
        :param validate: whether to validate df against the expected schema.
        :return: the fitted calibration slope, or NaN if undefined/unreliable.
        """
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        fit = self._get_fit(df, group_filter, mask, validate)
        return fit.slope if fit is not None else np.nan

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
        Analytical Wald confidence interval for the calibration slope,
        derived from the logistic recalibration fit's covariance matrix.

        :param df: dataframe containing y_true and y_pred_prob columns.
        :param group_filter: identifies the group to evaluate on.
        :param group_mask: alternative to group_filter; see __call__.
        :param ci_alpha: confidence level, e.g. 0.95 for a 95% CI.
        :param validate: whether to validate df against the expected schema.
        :param return_val: if True, also return the point estimate.
        :return: (low, high) CI bounds, or (value, (low, high)) if return_val.
        """
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        fit = self._get_fit(df, group_filter, mask, validate)

        if fit is None:
            val, ci = np.nan, (np.nan, np.nan)
        else:
            val = fit.slope
            ci = wald_ci(fit.slope, fit.cov_params[1, 1], ci_alpha=ci_alpha)

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
        Analytical variance of the calibration slope, read off the logistic
        recalibration fit's covariance matrix.

        :param df: dataframe containing y_true and y_pred_prob columns.
        :param group_filter: identifies the group to evaluate on.
        :param group_mask: alternative to group_filter; see __call__.
        :param validate: whether to validate df against the expected schema.
        :param return_val: if True, also return the point estimate.
        :return: the variance, or (value, variance) if return_val.
        """
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        fit = self._get_fit(df, group_filter, mask, validate)

        if fit is None:
            val, var = np.nan, np.nan
        else:
            val, var = fit.slope, fit.cov_params[1, 1]

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
        slopes, shared with DRMSCE/CalibrationIntercept's supporting curve
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
