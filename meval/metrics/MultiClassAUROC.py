from typing import Optional, Literal, cast
import pandas as pd

from ._metrics import auroc
from .ComparisonMetric import ComparisonMetric
from ..group_filter import GroupFilter


class MultiClassAUROC(ComparisonMetric):

    def __init__(self, test: bool = False, auroc_type: Literal['ovo', 'OvO', 'OVO', 'ovr', 'OvR', 'OVR'] = 'ovo'):
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

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        ) -> float | tuple[float, float] | tuple[float, tuple[float, float]]:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        y_true = self.get_multiclass_y_true(df, mask=mask, validate=validate)
        y_pred_prob = self.get_multiclass_y_pred_prob(df, mask=mask, validate=validate)
        return auroc(y_true.to_numpy(), y_pred_prob.to_numpy(), multiclass_mode=self.multiclass_mode)
