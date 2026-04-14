from abc import ABC, abstractmethod
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import pandera.pandas as pa
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from ..group_filter import GroupFilter
from .._array_types import MaskLike


class ComparisonMetric(ABC):

    y_true_cols = ['y_true', 'label', 'y', 'target']
    y_pred_prob_cols = ['y_pred_prob', 'y_prob']
    y_pred_cols = ['y_pred'] + y_pred_prob_cols
    y_float_pred_cols = ['y_pred']
    
    # multiclass case: we don't know how many classes there are; check for the right format of at least class 0
    y_pred_prob_cols_mc = ['y_pred_prob_0', 'y_prob_0']

    req_cols: Sequence[Sequence[str] | str]
    metric_name: str
    reference_class: str
    needs_all_classes: bool
    is_descriptive: bool
    test: bool

    def __init__(
            self,
            req_cols: Sequence[Sequence[str] | str],
            metric_name: str,
            reference_class: str,
            needs_all_classes: bool,
            is_descriptive: bool,
            test: bool = False
    ):
        self.req_cols = req_cols
        self.metric_name = metric_name
        self.reference_class = reference_class
        self.needs_all_classes = needs_all_classes
        self.is_descriptive = is_descriptive
        self.test = test

    @abstractmethod
    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        raise NotImplementedError

    def resolve_metrics(self, df: pd.DataFrame, validate: bool = True) -> list["ComparisonMetric"]:
        # Most metrics map to one scalar output column. Metrics that need to
        # expand into multiple scalar outputs (for example one per class) can
        # override this method.
        return [self]
    
    @staticmethod
    def _get_group_mask(
            df: pd.DataFrame, 
            group_filter: Optional[GroupFilter] = None, 
            mask: Optional[MaskLike] = None,
            validate: bool = True
            ) -> MaskLike:

        if mask is None:
            if group_filter is None:
                mask = pd.Series([True] * len(df.index), index=df.index)
            else:
                mask = group_filter(df, validate=validate)

        return mask

    @staticmethod
    def _resolve_col_name(df: pd.DataFrame, candidate_cols: Sequence[str], col_desc: str) -> str:
        # Resolve the first recognized column name for a semantic field (e.g. y_true, y_pred).
        # This centralizes alias handling and keeps error messages consistent across helpers.
        matches = [col for col in candidate_cols if col in df.columns]
        if not matches:
            raise RuntimeError(f"Found no {col_desc} column in the provided dataframe. Recognized column names are: {list(candidate_cols)}.")
        return matches[0]

    @staticmethod
    def _masked_series(col: pd.Series, mask: Optional[MaskLike]) -> pd.Series:
        if mask is None:
            return col

        if isinstance(mask, pd.Series) and not pd.api.types.is_bool_dtype(mask.dtype):
            raise TypeError("Pandas mask series must have boolean dtype.")

        if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.integer):
            return pd.Series(col.to_numpy(copy=False)[mask], copy=False)

        # Fast path for aligned boolean masks avoids pandas internals-heavy
        # boolean indexing on each call in hot loops.
        if isinstance(mask, np.ndarray) and mask.dtype == bool and len(mask) == len(col):
            return pd.Series(col.to_numpy(copy=False)[mask], copy=False)

        if (
            isinstance(mask, pd.Series)
            and mask.dtype == bool
            and len(mask) == len(col)
            and (mask.index is col.index or mask.index.equals(col.index))
        ):
            mask_np = mask.to_numpy(copy=False)
            return pd.Series(col.to_numpy(copy=False)[mask_np], copy=False)

        # Fallback keeps pandas' full index-alignment semantics.
        return col[mask]

    @staticmethod
    def _masked_array(col: pd.Series, mask: Optional[MaskLike]) -> np.ndarray:
        if mask is None:
            return col.to_numpy(copy=False)

        if isinstance(mask, pd.Series) and not pd.api.types.is_bool_dtype(mask.dtype):
            raise TypeError("Pandas mask series must have boolean dtype.")

        if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.integer):
            return col.to_numpy(copy=False)[mask]

        if isinstance(mask, np.ndarray) and mask.dtype == bool and len(mask) == len(col):
            return col.to_numpy(copy=False)[mask]

        if (
            isinstance(mask, pd.Series)
            and mask.dtype == bool
            and len(mask) == len(col)
            and (mask.index is col.index or mask.index.equals(col.index))
        ):
            return col.to_numpy(copy=False)[mask.to_numpy(copy=False)]

        return col[mask].to_numpy(copy=False)

    @staticmethod
    def _masked_frame(df: pd.DataFrame, mask: Optional[MaskLike]) -> pd.DataFrame:
        if mask is None:
            return df

        if isinstance(mask, pd.Series) and not pd.api.types.is_bool_dtype(mask.dtype):
            raise TypeError("Pandas mask series must have boolean dtype.")

        if isinstance(mask, np.ndarray) and np.issubdtype(mask.dtype, np.integer):
            return df.iloc[mask]

        return df.loc[mask]

    @staticmethod
    def _validate_bool_array(arr: np.ndarray) -> None:
        assert np.issubdtype(arr.dtype, np.bool_), "Expected boolean array."

    @staticmethod
    def _validate_int_array(arr: np.ndarray) -> None:
        assert np.issubdtype(arr.dtype, np.integer), "Expected integer array."

    @staticmethod
    def _validate_float_array(arr: np.ndarray) -> None:
        assert np.issubdtype(arr.dtype, np.floating), "Expected float array."
        assert not np.any(np.isnan(arr)), "Expected non-null float values."

    @staticmethod
    def _validate_prob_array(arr: np.ndarray) -> None:
        ComparisonMetric._validate_float_array(arr)
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0), "Expected probabilities in [0, 1]."

    def get_group_mask(
            self, 
            df: pd.DataFrame, 
            group_filter: Optional[GroupFilter] = None, 
            group_mask: Optional[MaskLike] = None,
            validate: bool = True
            ) -> MaskLike:
        if validate:
            for req_col in self.req_cols:
                if isinstance(req_col, list):  # at least one of these must be present
                    assert any([col in df.columns for col in req_col]), f"Expected one of {req_col} in df.columns."
                else:
                    assert req_col in df.columns, f"Expected {req_col} in df.columns."

        return ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=group_mask, validate=validate)
    
    @staticmethod
    def get_y_true(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.Series | np.ndarray:

        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)

        colname = ComparisonMetric._resolve_col_name(df, ComparisonMetric.y_true_cols, "y_true")
        if return_array:
            return ComparisonMetric._masked_array(df[colname], mask)

        y_true = ComparisonMetric._masked_series(df[colname], mask)

        return y_true

    @staticmethod
    def get_binary_y_true(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.Series | np.ndarray:
        y_true = ComparisonMetric.get_y_true(
            df,
            group_filter=group_filter,
            mask=mask,
            validate=validate,
            return_array=return_array,
        )

        if return_array:
            assert isinstance(y_true, np.ndarray)
            y_true_np = y_true
            if validate:
                ComparisonMetric._validate_bool_array(y_true_np)
            return y_true_np
        else:
            assert isinstance(y_true, pd.Series)
            y_true_ser = y_true
            if validate:
                pa.SeriesSchema(bool, nullable=False, unique=False).validate(y_true_ser)

            return y_true_ser  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a bool series
    
    @staticmethod
    def get_float_y_true(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.Series | np.ndarray:
        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)
        colname = ComparisonMetric._resolve_col_name(df, ComparisonMetric.y_true_cols, "y_true")

        if return_array:
            y_true_np = ComparisonMetric._masked_array(df[colname], mask).astype(float, copy=False)
            if validate:
                ComparisonMetric._validate_float_array(y_true_np)
            return y_true_np
        else:
            y_true = ComparisonMetric._masked_series(df[colname], mask)

            if validate:
                pa.SeriesSchema(float, nullable=False, unique=False).validate(y_true)

            return y_true

    @staticmethod
    def get_multiclass_y_true(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.Series | np.ndarray:
        y_true = ComparisonMetric.get_y_true(
            df,
            group_filter=group_filter,
            mask=mask,
            validate=validate,
            return_array=return_array,
        )

        if return_array:
            assert isinstance(y_true, np.ndarray)
            y_true_np = y_true
            if validate:
                ComparisonMetric._validate_int_array(y_true_np)
            return y_true_np
        else:
            assert isinstance(y_true, pd.Series)
            y_true_ser = y_true

            if validate:
                pa.SeriesSchema(int, nullable=False, unique=False).validate(y_true_ser)

            return y_true_ser  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a float series

    @staticmethod
    def get_binary_y_pred_prob(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.Series | np.ndarray:

        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)
        colname = ComparisonMetric._resolve_col_name(df, ComparisonMetric.y_pred_prob_cols, "pred_prob")
        if return_array:
            y_pred_prob_np = ComparisonMetric._masked_array(df[colname], mask).astype(float, copy=False)
            if validate:
                ComparisonMetric._validate_prob_array(y_pred_prob_np)
            return y_pred_prob_np
        else:
            y_pred_prob = ComparisonMetric._masked_series(df[colname], mask)

            if validate:
                pa.SeriesSchema(float, 
                                checks=[pa.Check.greater_than_or_equal_to(0.0),
                                        pa.Check.less_than_or_equal_to(1.0)],
                                nullable=False, 
                                unique=False
                                ).validate(y_pred_prob)

            return y_pred_prob  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a float series

    @staticmethod
    def _get_sorted_multiclass_pred_prob_cols(df: pd.DataFrame) -> list[str]:
        # Find all y_pred_prob columns (format: y_pred_prob_{cls_id}).
        y_pred_prob_cols = [col for col in df.columns if col.startswith("y_pred_prob_")]

        if len(y_pred_prob_cols) == 0:
            raise RuntimeError("Found no pred_prob columns in the provided dataframe. Expected columns of the form 'y_pred_prob_{cls_id}'.")

        class_ids = []
        for col in y_pred_prob_cols:
            try:
                cls_id = int(col.replace("y_pred_prob_", ""))
                class_ids.append(cls_id)
            except ValueError:
                raise RuntimeError(f"Invalid column name '{col}'. Expected format 'y_pred_prob_{{cls_id}}' where cls_id is an integer.")

        return [f"y_pred_prob_{cls_id}" for cls_id in sorted(class_ids)]

    @staticmethod
    def get_multiclass_y_pred_prob(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.DataFrame | np.ndarray:

        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)

        sorted_cols = ComparisonMetric._get_sorted_multiclass_pred_prob_cols(df)
        
        # Extract the dataframe with masked rows.
        y_pred_prob = ComparisonMetric._masked_frame(df[sorted_cols], mask)

        if return_array:
            y_pred_prob_np = y_pred_prob.to_numpy(dtype=float, copy=False)
            if validate:
                ComparisonMetric._validate_prob_array(y_pred_prob_np)
                row_sums = y_pred_prob_np.sum(axis=1)
                assert np.allclose(row_sums, 1.0, atol=1e-6), "Predicted probabilities must sum to 1 for each sample"
            return y_pred_prob_np
        else:
            if validate:
                # Validate that all columns are float and in [0, 1]
                for col in sorted_cols:
                    pa.SeriesSchema(float, 
                                    checks=[pa.Check.greater_than_or_equal_to(0.0),
                                            pa.Check.less_than_or_equal_to(1.0)],
                                    nullable=False, 
                                    unique=False
                                    ).validate(y_pred_prob[col])
                
                # Optionally: validate that probabilities sum to 1 (or close to it) for each row
                row_sums = y_pred_prob.sum(axis=1)
                assert np.allclose(row_sums, 1.0, atol=1e-6), "Predicted probabilities must sum to 1 for each sample"

            return y_pred_prob  # type: ignore

    def get_binary_y_pred(
            self, 
            df: pd.DataFrame, 
            group_filter: Optional[GroupFilter] = None,
            mask: Optional[MaskLike] = None,
            validate: bool = True,
            return_array: bool = False,
            ) -> pd.Series | np.ndarray:

        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)

        if 'y_pred' in df.columns:
            y_pred = ComparisonMetric._masked_series(df['y_pred'], mask)

        elif isinstance(self, ThresholdedComparisonMetric) and self.threshold is not None:
            y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate, return_array=False)
            assert isinstance(y_pred_prob, pd.Series)
            y_pred = y_pred_prob >= self.threshold

        else:
            raise RuntimeError

        if validate:
            pa.SeriesSchema(bool, nullable=False, unique=False).validate(y_pred)

        if return_array:
            return y_pred.to_numpy(dtype=bool, copy=False)

        return y_pred  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a bool series

    @staticmethod
    def get_multiclass_y_pred(
            df: pd.DataFrame,
            group_filter: Optional[GroupFilter] = None,
            mask: Optional[MaskLike] = None,
            validate: bool = True,
            return_array: bool = False,
            ) -> pd.Series | np.ndarray:

        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)

        if 'y_pred' in df.columns:
            if return_array:
                y_pred_np = np.asarray(ComparisonMetric._masked_array(df['y_pred'], mask))
                if validate:
                    ComparisonMetric._validate_int_array(y_pred_np)
                return y_pred_np.astype(int, copy=False)

            y_pred = ComparisonMetric._masked_series(df['y_pred'], mask)

        else:
            has_mc_prob_cols = any(col.startswith("y_pred_prob_") for col in df.columns)
            if not has_mc_prob_cols:
                raise RuntimeError("Expected multiclass predictions in either y_pred or y_pred_prob_{cls_id} columns.")
            
            y_pred_prob = ComparisonMetric.get_multiclass_y_pred_prob(
                df,
                mask=mask,
                validate=validate,
                return_array=True,
            )
            assert isinstance(y_pred_prob, np.ndarray)
            class_cols = ComparisonMetric._get_sorted_multiclass_pred_prob_cols(df)
            class_ids = np.array([int(col.replace("y_pred_prob_", "")) for col in class_cols], dtype=int)
            y_pred_np = class_ids[np.argmax(y_pred_prob, axis=1)]

            if return_array:
                if validate:
                    ComparisonMetric._validate_int_array(y_pred_np)
                return y_pred_np

            y_pred = pd.Series(y_pred_np, copy=False)

        if validate:
            pa.SeriesSchema(int, nullable=False, unique=False).validate(y_pred)

        return y_pred  # type: ignore
    
    @staticmethod
    def get_float_y_pred(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_array: bool = False,
        ) -> pd.Series | np.ndarray:
        mask = ComparisonMetric._get_group_mask(df, group_filter=group_filter, mask=mask, validate=validate)
        colname = ComparisonMetric._resolve_col_name(df, ComparisonMetric.y_float_pred_cols, "y_pred")

        if return_array:
            y_pred_np = ComparisonMetric._masked_array(df[colname], mask).astype(float, copy=False)
            if validate:
                ComparisonMetric._validate_float_array(y_pred_np)
            return y_pred_np
        else:
            y_pred = ComparisonMetric._masked_series(df[colname], mask)

            if validate:
                pa.SeriesSchema(float, nullable=False, unique=False).validate(y_pred)

            return y_pred

    def __repr__(self) -> str:
        return self.metric_name
    
    @property
    def metric_name_low(self) -> str:
        return self.metric_name + ' (low)'

    @property
    def metric_name_med(self) -> str:
        return self.metric_name + ' (med)'
    
    @property
    def metric_name_high(self) -> str:
        return self.metric_name + ' (high)'
    
    @property
    def metric_name_pval(self) -> str:
        return self.metric_name + ' pval'
    
    @property
    def metric_name_effect(self) -> str:
        return self.metric_name + ' effect'    
    

class CurveBasedComparisonMetric(ComparisonMetric):

    def __init__(
            self,
            req_cols: Sequence[Sequence[str] | str],
            metric_name: str,
            reference_class: str,
            needs_all_classes: bool,
            is_descriptive: bool,
            test: bool = False
    ):
        super().__init__(req_cols=req_cols,
                         metric_name=metric_name,
                         reference_class=reference_class,
                         needs_all_classes=needs_all_classes,
                         is_descriptive=is_descriptive,
                         test=test)

    @abstractmethod
    def plot_supporting_curve(
        self, 
        metric_results_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        group_colors_dict: Optional[dict[str, str]] = None,
        add_all_group: bool = True,
        threshold: Optional[float] = None
        ) -> tuple[go.Figure, list[BaseTraceType], dict]:
        raise NotImplementedError
    

class MetricWithAnalyticalCI(ComparisonMetric):

    @abstractmethod
    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_ci: bool = False
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_ci(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        y_true: Optional[pd.Series | np.ndarray] = None,
        y_pred: Optional[pd.Series | np.ndarray] = None,
        y_pred_prob: Optional[pd.Series | np.ndarray] = None,
        return_val: Optional[bool] = False
        ) -> tuple[float, float] | tuple[int, tuple[float, float]] | tuple[float, tuple[float, float]]:
        raise NotImplementedError
    

class MetricWithAnalyticalVar(ComparisonMetric):

    @abstractmethod
    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        validate: bool = True,
        y_true: Optional[pd.Series | np.ndarray] = None,
        y_pred: Optional[pd.Series | np.ndarray] = None,
        y_pred_prob: Optional[pd.Series | np.ndarray] = None,
        return_val: Optional[bool] = False
        ) -> float | tuple[float, float] | tuple[int, float]:
        raise NotImplementedError


class ThresholdedComparisonMetric(ComparisonMetric):

    threshold: Optional[float] = None

    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.set_threshold(threshold)

    def set_threshold(self, threshold: Optional[float]):

        if threshold is not None:
            assert self.threshold is None

        self.threshold = threshold

        if threshold is not None:
            self.metric_name = f'{self.metric_name} (thr={threshold:.2g})'

