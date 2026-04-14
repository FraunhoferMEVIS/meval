from typing import Optional
import pandas as pd

import numpy as np
from .ComparisonMetric import ComparisonMetric, MaskLike
from ..group_filter import GroupFilter

class Count(ComparisonMetric):

    def __init__(self, test: bool = False):
        super().__init__(
            req_cols=[],
            metric_name='Count',
            needs_all_classes=False,
            reference_class='self',
            is_descriptive=True,
            test=test
        )

    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True
        ) -> int:
        
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.integer):
            return len(mask)
        return int(mask.sum())



