
import numpy as np
import pandas as pd

from ..group_filter import GroupFilter
from ._metrics import specificity
from .ComparisonMetric import ComparisonMetric, DecompositionTerm, MaskLike, ThresholdedComparisonMetric


class Specificity(ThresholdedComparisonMetric):
    # = 1 - FPR

    def __init__(self, threshold: float | None = None, test: bool = False):

        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_cols],
            metric_name='Spec',
            threshold=threshold,
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            direction="maximize",
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None, 
        group_mask: MaskLike | None = None,
        validate: bool = True
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True)

        return specificity(y_true, y_pred)

    def decompose(self, df: pd.DataFrame, validate: bool = True) -> list[DecompositionTerm] | None:
        # TN / (TN + FP): the share of negatives that are predicted negative.
        mask = self.get_group_mask(df, validate=validate)
        y_true = np.asarray(self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True), dtype=bool)
        y_pred = np.asarray(self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True), dtype=bool)
        return [DecompositionTerm(1.0, (~y_pred).astype(float), ~y_true)]



