from typing import Optional
import numpy as np
import pandas as pd

from ._metrics import recall
from .ComparisonMetric import ComparisonMetric, ThresholdedComparisonMetric, MaskLike
from ..group_filter import GroupFilter


class Recall(ThresholdedComparisonMetric):

    def __init__(self, threshold: Optional[float] = None, test: bool = False, per_class: bool = False):

        self.per_class = per_class

        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_cols],
            metric_name='Rec',
            threshold=threshold,
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            test=test
        )

    def resolve_metrics(self, df: pd.DataFrame, validate: bool = True) -> list[ComparisonMetric]:
        if not self.per_class:
            y_true = self.get_y_true(df, validate=validate, return_array=True)
            assert isinstance(y_true, np.ndarray)
            if np.issubdtype(y_true.dtype, np.bool_):
                return [self]
            if np.issubdtype(y_true.dtype, np.integer):
                raise NotImplementedError("Recall(per_class=False) is currently only implemented for binary y_true.")
            raise RuntimeError("y_true must be bool (binary) or integer (multiclass).")

        y_true = self.get_y_true(df, validate=validate, return_array=True)
        assert isinstance(y_true, np.ndarray)
        if np.issubdtype(y_true.dtype, np.bool_):
            raise NotImplementedError("Recall(per_class=True) with binary y_true is not implemented.")
        if not np.issubdtype(y_true.dtype, np.integer):
            raise RuntimeError("y_true must be bool (binary) or integer (multiclass).")

        class_ids = sorted(int(x) for x in np.unique(y_true))
        return [_PerClassRecall(class_id=class_id, test=self.test) for class_id in class_ids]

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        y_pred = self.get_binary_y_pred(df, mask=mask, validate=validate, return_array=True)
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)

        return recall(y_true, y_pred)


class _PerClassRecall(ComparisonMetric):

    def __init__(self, class_id: int, test: bool = False):
        self.class_id = class_id
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ["y_pred", *ComparisonMetric.y_pred_prob_cols_mc]],
            metric_name=f'Rec[class={class_id}]',
            reference_class='self',
            needs_all_classes=False,
            is_descriptive=False,
            test=test
        )

    def __call__(
        self,
        df: pd.DataFrame,
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        validate: bool = True
        ) -> float:

        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true_mc = self.get_multiclass_y_true(df, mask=mask, validate=validate, return_array=True)
        assert isinstance(y_true_mc, np.ndarray)
        pred_class = self.get_multiclass_y_pred(df, mask=mask, validate=validate, return_array=True)
        assert isinstance(pred_class, np.ndarray)

        y_true_bin = y_true_mc == self.class_id
        y_pred_bin = pred_class == self.class_id

        return recall(y_true_bin, y_pred_bin)




