from typing import Optional
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go
import pandas as pd

from ._metrics import area_under_precision_recall_gain_score
from .ComparisonMetric import ComparisonMetric, CurveBasedComparisonMetric
from ..diags import prg_diag
from ..group_filter import GroupFilter
from ..select_groups import select_extreme_groups


class AUPRG(CurveBasedComparisonMetric):
    # Area under the precision-recall-gain curve, see
    # https://research-information.bris.ac.uk/ws/portalfiles/portal/72164009/5867_precision_recall_gain_curves_pr_analysis_done_right.pdf 
    # and https://dl.acm.org/doi/10.1145/3593013.3594045
    # Recommended discrimination metric in the (strongly) class-imbalanced case
    # Also see https://stats.stackexchange.com/a/625752/131402 for a related discussion

    def __init__(self, rec_gain_min: float = 0, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name="AUPRG"
            if rec_gain_min == 0
            else "pAUPRG(recgmin=" + str(rec_gain_min) + ")",
            reference_class="self",  # ?????? https://proceedings.neurips.cc/paper_files/paper/2019/file/73e0f7487b8e5297182c5a711d20bf26-Paper.pdf
            needs_pos_and_neg=True,
            is_descriptive=False,
            test=test
        )
        self.rec_gain_min = rec_gain_min

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
        #return calc_auprg_from_data(y_true.to_numpy(), y_pred_prob.to_numpy())
        return area_under_precision_recall_gain_score(y_true.to_numpy(), y_pred_prob.to_numpy(), 
                                                      rec_gain_min=self.rec_gain_min)

    def plot_supporting_curve(
            self, 
            metric_results_df: pd.DataFrame, 
            test_df: pd.DataFrame, 
            group_colors_dict: Optional[dict[str, str]] = None, 
            add_all_group: bool = True,
            threshold: Optional[float] = None,
            ) -> tuple[go.Figure, list[BaseTraceType], dict]:
        
        plot_groups_prg = select_extreme_groups(metric_results_df, 
                                                cols={self.metric_name_low: False, self.metric_name_high: True},
                                                n_groups=3, add_all_group=add_all_group, sort=True)
        fig, prg_traces, layout_kwargs = prg_diag(test_df, plot_groups_prg, group_color_dict=group_colors_dict, legend=False, fig_title='',
                                                  rec_gain_min=self.rec_gain_min, threshold=threshold)
        return fig, prg_traces, layout_kwargs