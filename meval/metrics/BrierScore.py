import numpy as np
import pandas as pd

from ..group_filter import GroupFilter
from ._metrics import brier_scores
from .ComparisonMetric import ComparisonMetric, DecompositionTerm, MaskLike


class BrierScore(ComparisonMetric):

    def __init__(self, balanced: bool = True, test: bool = False):
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols],
            metric_name='BS (bal)' if balanced else 'BS',
            reference_class='self',
            needs_all_classes=bool(balanced),
            is_descriptive=False,
            direction="minimize",
            test=test
        )
        self.balanced = balanced

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None, 
        group_mask: MaskLike | None = None,
        validate: bool = True
        ) -> float:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)

        y_true = np.asarray(self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True), dtype=bool)
        y_pred_prob = np.asarray(self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True), dtype=float)

        brier_score, _brier_score_pos, _brier_score_neg, brier_score_bal = brier_scores(y_true, y_pred_prob)
        if self.balanced:
            return brier_score_bal
        else:
            return brier_score

    def decompose(self, df: pd.DataFrame, validate: bool = True) -> list[DecompositionTerm] | None:
        mask = self.get_group_mask(df, validate=validate)
        y_true = np.asarray(self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True), dtype=bool)
        y_pred_prob = np.asarray(self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=True), dtype=float)
        squared_error = np.square(y_true - y_pred_prob)
        if self.balanced:
            # 0.5 * Brier score of the positives + 0.5 * Brier score of the negatives
            return [DecompositionTerm(0.5, squared_error, y_true), DecompositionTerm(0.5, squared_error, ~y_true)]
        return [DecompositionTerm(1.0, squared_error, np.ones(len(y_true), dtype=bool))]



