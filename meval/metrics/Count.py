
import numpy as np
import pandas as pd

from ..group_filter import GroupFilter
from .ComparisonMetric import ComparisonMetric, MaskLike


class Count(ComparisonMetric):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[],
            metric_name='Count',
            needs_all_classes=False,
            reference_class='self',
            is_descriptive=True,
            direction="N/A",
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: GroupFilter | None = None, 
        group_mask: MaskLike | None = None,
        validate: bool = True
        ) -> int:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.integer):
            return len(mask)
        return int(mask.sum())



