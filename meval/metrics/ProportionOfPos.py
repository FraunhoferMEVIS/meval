from typing import Optional
import numpy as np
import pandas as pd

from .ComparisonMetric import ComparisonMetric, MaskLike
from ..group_filter import GroupFilter


class ProportionOfPos(ComparisonMetric):

    def __init__(self, test: bool = False, per_class: bool = False):
        self.per_class = per_class
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols],
            metric_name='p(y=1|G=g)',
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
                raise NotImplementedError("ProportionOfPos(per_class=False) is currently only implemented for binary y_true.")
            raise RuntimeError("y_true must be bool (binary) or integer (multiclass).")

        y_true = self.get_y_true(df, validate=validate, return_array=True)
        assert isinstance(y_true, np.ndarray)
        if np.issubdtype(y_true.dtype, np.bool_):
            raise NotImplementedError("ProportionOfPos(per_class=True) with binary y_true is not implemented.")
        if not np.issubdtype(y_true.dtype, np.integer):
            raise RuntimeError("y_true must be bool (binary) or integer (multiclass).")

        class_ids = sorted(int(x) for x in np.unique(y_true))
        return [_PerClassProportionOfPos(class_id=class_id, test=self.test) for class_id in class_ids]

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True
        ) -> float:
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_binary_y_true(df, mask=mask, validate=validate, return_array=True)
        return float(y_true.mean())


class _PerClassProportionOfPos(ComparisonMetric):

    def __init__(self, class_id: int, test: bool = False):
        self.class_id = class_id
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols],
            metric_name=f'p(y={class_id}|G=g)',
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
        return float((y_true_mc == self.class_id).mean())





