from typing import Optional, Literal, cast
import pandas as pd
import numpy as np

from ._metrics import auroc
from .ComparisonMetric import ComparisonMetric, MaskLike
from .fastauc import fast_ovo_auc_numba
from ..config import settings
from ..group_filter import GroupFilter


class MultiClassAUROC(ComparisonMetric):

    def __init__(self, test: bool = False, 
                 auroc_type: Literal['ovo', 'OvO', 'OVO', 'ovr', 'OvR', 'OVR'] = 'ovo',
                 min_samples_per_class: Optional[int] = None):
        # Default to OvO (instead of OvR) because that is better comparable between subgroups with differing label distributions.
        assert auroc_type in ["ovo", "ovr"]
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols, ComparisonMetric.y_pred_prob_cols_mc],
            metric_name="OvR-AUROC" if auroc_type.lower() == "ovr" else "OvO-AUROC",
            reference_class="self",  # ?????? https://proceedings.neurips.cc/paper_files/paper/2019/file/73e0f7487b8e5297182c5a711d20bf26-Paper.pdf
            needs_all_classes=True if auroc_type.lower() == "ovo" else False,
            is_descriptive=False,
            test=test
        )
        self.multiclass_mode: Literal['ovo', 'ovr'] = cast(Literal['ovo', 'ovr'], auroc_type.lower())
        self.min_samples_per_class: int = settings.auroc_min_cases_per_class if min_samples_per_class is None else min_samples_per_class

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        ) -> float | tuple[float, float] | tuple[float, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true_np = np.asarray(self.get_multiclass_y_true(df, mask=mask, validate=validate, return_array=True), dtype=int)
        y_pred_prob_np = np.asarray(self.get_multiclass_y_pred_prob(df, mask=mask, validate=validate, return_array=True), dtype=float)

        if self.multiclass_mode == "ovo":
            classes, counts = np.unique(y_true_np, return_counts=True)

            # Preserve previous behavior: OvO requires all modeled classes present,
            # and each present class needs enough samples.
            if len(classes) < y_pred_prob_np.shape[1] or np.any(counts < self.min_samples_per_class):
                return np.nan

            # Rebuild class IDs from sorted prob-column names so label IDs stay
            # in the same order as y_pred_prob_np columns passed to fast_ovo_auc_numba.
            sorted_cols = ComparisonMetric._get_sorted_multiclass_pred_prob_cols(df)
            labels = [int(col.replace("y_pred_prob_", "")) for col in sorted_cols]
            return fast_ovo_auc_numba(y_true_np, y_pred_prob_np, labels=labels)

        return auroc(y_true_np, y_pred_prob_np,
                     multiclass_mode=self.multiclass_mode, min_cases_per_class=self.min_samples_per_class)



