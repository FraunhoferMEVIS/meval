from abc import ABC, abstractmethod
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import pandera.pandas as pa
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from ..group_filter import GroupFilter


class ComparisonMetric(ABC):

    y_true_cols = ['y_true', 'label', 'y', 'target']
    y_pred_prob_cols = ['y_pred_prob', 'y_prob']
    y_pred_cols = ['y_pred'] + y_pred_prob_cols
    
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
        group_mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        raise NotImplementedError
    
    def get_group_mask(
            self, 
            df: pd.DataFrame, 
            group_filter: Optional[GroupFilter] = None, 
            group_mask: Optional[pd.Series] = None,
            validate: bool = True
            ) -> pd.Series:
        if validate:
            for req_col in self.req_cols:
                if isinstance(req_col, list):  # at least one of these must be present
                    assert any([col in df.columns for col in req_col]), f"Expected one of {req_col} in df.columns."
                else:
                    assert req_col in df.columns, f"Expected {req_col} in df.columns."

        if group_mask is None:
            assert group_filter is not None
            group_mask = group_filter(df, validate=validate)
        
        return group_mask

    @staticmethod
    def get_binary_y_true(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> pd.Series:

        if mask is None:
            if group_filter is None:
                mask = pd.Series([True] * len(df.index), index=df.index)
            else:
                mask = group_filter(df, validate=validate)

        for colname in ComparisonMetric.y_true_cols:
            if colname in df.columns:
                y_true = df[colname][mask]
                break
        else:
            raise RuntimeError(f"Found no label column in the provided dataframe. Recognized column names are: {ComparisonMetric.y_true_cols}.")

        if validate:
            pa.SeriesSchema(bool, nullable=False, unique=False).validate(y_true)

        return y_true  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a bool series
    
    @staticmethod
    def get_multiclass_y_true(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> pd.Series:

        if mask is None:
            if group_filter is None:
                mask = pd.Series([True] * len(df.index), index=df.index)
            else:
                mask = group_filter(df, validate=validate)

        for colname in ComparisonMetric.y_true_cols:
            if colname in df.columns:
                y_true = df[colname][mask]
                break
        else:
            raise RuntimeError(f"Found no label column in the provided dataframe. Recognized column names are: {ComparisonMetric.y_true_cols}.")

        if validate:
            pa.SeriesSchema(int, nullable=False, unique=False).validate(y_true)

        return y_true  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be an int series

    @staticmethod
    def get_binary_y_pred_prob(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> pd.Series:

        if mask is None:
            if group_filter is None:
                mask = pd.Series([True] * len(df.index), index=df.index)
            else:
                mask = group_filter(df, validate=validate)

        for colname in ComparisonMetric.y_pred_prob_cols:
            if colname in df.columns:
                y_pred_prob = df[colname][mask]
                break
        else:
            raise RuntimeError(f"Found no pred_prob column in the provided dataframe. Recognized column names are: {ComparisonMetric.y_pred_prob_cols}.")

        if validate:
            pa.SeriesSchema(float, 
                            checks=[pa.Check.greater_than_or_equal_to(0.0),
                                    pa.Check.less_than_or_equal_to(1.0)],
                            nullable=False, 
                            unique=False
                            ).validate(y_pred_prob)

        return y_pred_prob  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a float series

    @staticmethod
    def get_multiclass_y_pred_prob(
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        mask: Optional[pd.Series] = None,
        validate: bool = True
        ) -> pd.DataFrame:

        if mask is None:
            if group_filter is None:
                mask = pd.Series([True] * len(df.index), index=df.index)
            else:
                mask = group_filter(df, validate=validate)

        # Find all y_pred_prob columns (format: y_pred_prob_{cls_id})
        y_pred_prob_cols = [col for col in df.columns if col.startswith("y_pred_prob_")]
        
        if len(y_pred_prob_cols) == 0:
            raise RuntimeError("Found no pred_prob columns in the provided dataframe. Expected columns of the form 'y_pred_prob_{cls_id}'.")
        
        # Extract class IDs and sort to ensure consistent ordering
        class_ids = []
        for col in y_pred_prob_cols:
            try:
                cls_id = int(col.replace("y_pred_prob_", ""))
                class_ids.append(cls_id)
            except ValueError:
                raise RuntimeError(f"Invalid column name '{col}'. Expected format 'y_pred_prob_{{cls_id}}' where cls_id is an integer.")
        
        # Sort columns by class ID
        sorted_cols = [f"y_pred_prob_{cls_id}" for cls_id in sorted(class_ids)]
        
        # Extract the dataframe with masked rows
        y_pred_prob = df.loc[mask, sorted_cols]
        
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
            mask: Optional[pd.Series] = None,
            validate: bool = True,
            ) -> pd.Series:

        if mask is None:
            if group_filter is None:
                mask = pd.Series([True] * len(df.index), index=df.index)
            else:
                mask = group_filter(df, validate=validate)

        if 'y_pred' in df.columns:
            y_pred = df['y_pred'][mask]

        elif isinstance(self, ThresholdedComparisonMetric) and self.threshold is not None:
            y_pred_prob = self.get_binary_y_pred_prob(df, mask=mask, validate=validate)
            y_pred = y_pred_prob >= self.threshold

        else:
            raise RuntimeError

        if validate:
            pa.SeriesSchema(bool, nullable=False, unique=False).validate(y_pred)

        return y_pred  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a bool series
    
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
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        return_ci: bool = False
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_ci(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[pd.Series] = None,
        ci_alpha: float = 0.95,
        validate: bool = True,
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
        y_pred_prob: Optional[pd.Series] = None,        
        return_val: Optional[bool] = False
        ) -> tuple[float, float] | tuple[int, tuple[float, float]] | tuple[float, tuple[float, float]]:
        raise NotImplementedError
    

class MetricWithAnalyticalVar(ComparisonMetric):

    @abstractmethod
    def __call__(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None, 
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        return_var: bool = False
        ) -> float | int | tuple[float, float] | tuple[int, float] | tuple[float, tuple[float, float]] | tuple[int, tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_variance(
        self, 
        df: pd.DataFrame, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[pd.Series] = None,
        validate: bool = True,
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
        y_pred_prob: Optional[pd.Series] = None,        
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
