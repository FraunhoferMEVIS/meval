from typing import Optional
import pandas as pd

from ._metrics import brier_scores
from .ComparisonMetric import ComparisonMetric
from ..group_filter import GroupFilter


class BrierScore(ComparisonMetric):

    def __init__(self, balanced: bool = True, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name='BS (bal)' if balanced else 'BS',
            reference_class='self',
            needs_pos_and_neg=True if balanced else False,
            is_descriptive=False,
            test=test
        )
        self.balanced = balanced


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

        brier_score, brier_score_pos, brier_score_neg, brier_score_bal = brier_scores(y_true.to_numpy(), y_pred_prob.to_numpy())
        if self.balanced:
            return brier_score_bal
        else:
            return brier_score