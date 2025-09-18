from typing import Optional
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go
import pandas as pd

from .ComparisonMetric import ComparisonMetric, CurveBasedComparisonMetric
from ._calibration import get_unbiased_calibration_rmse
from ..diags import rel_diag
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups

# Debiased root-mean-squared calibration error, see https://dl.acm.org/doi/10.1145/3593013.3594045
# To be preferred over ECE (expected calibration error) - which is commonly used to measure calibration - because that has (strong!) sample size biases,
# preventing between-group comparisons if the groups have different sizes. (And also just because ECE is an awful metric overall.)
# Notice the newer smECE is affected by sample size bias in the same way as ECE.
class DRMSCE(CurveBasedComparisonMetric):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name = 'DRMSCE',
            reference_class = 'self',
            needs_pos_and_neg = False,
            is_descriptive = False,
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> float:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate)
        y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate)
        return get_unbiased_calibration_rmse(y_true.to_numpy(), y_pred_prob.to_numpy())
    
    def plot_supporting_curve(
            self, 
            metric_results_df: pd.DataFrame, 
            test_df: pd.DataFrame, 
            group_colors_dict: Optional[dict[str, str]] = None, 
            add_all_group: bool = True,
            threshold: Optional[float] = None,
            ) -> tuple[go.Figure, list[BaseTraceType], dict]:
        
        plot_groups_rel = select_extreme_groups(metric_results_df, 
                                                cols={self.metric_name_low: False, self.metric_name_high: True},
                                                n_groups=3, add_all_group=add_all_group, sort=True)
        fig, rel_traces, layout_kwargs = rel_diag(test_df, plot_groups_rel, group_color_dict=group_colors_dict, add_risk_density=False,
                                                  legend=False, fig_title=None, threshold=threshold)
        return fig, rel_traces, layout_kwargs