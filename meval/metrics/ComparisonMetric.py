from abc import ABC, abstractmethod
from typing import Optional, Sequence
import pandas as pd
import pandera.pandas as pa
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from ..group_filter import GroupFilter


class ComparisonMetric(ABC):

    y_true_cols = ['y_true', 'label', 'y', 'target']
    y_pred_prob_cols = ['y_pred_prob', 'y_prob']
    y_pred_cols = ['y_pred'] + y_pred_prob_cols

    req_cols: Sequence[Sequence[str] | str]
    metric_name: str
    reference_class: str
    needs_pos_and_neg: bool
    is_descriptive: bool
    test: bool

    def __init__(
            self,
            req_cols: Sequence[Sequence[str] | str],
            metric_name: str,
            reference_class: str,
            needs_pos_and_neg: bool,
            is_descriptive: bool,
            test: bool = False
    ):
        self.req_cols = req_cols
        self.metric_name = metric_name
        self.reference_class = reference_class
        self.needs_pos_and_neg = needs_pos_and_neg
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
            raise RuntimeError(f"Found no binary label column in the provided dataframe. Recognized column names are: {ComparisonMetric.y_true_cols}.")

        if validate:
            pa.SeriesSchema(bool, nullable=False, unique=False).validate(y_true)

        return y_true  # type: ignore  - I don't know how to tell pandera that this thing is really guaranteed to be a bool series
    
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
            needs_pos_and_neg: bool,
            is_descriptive: bool,
            test: bool = False
    ):
        super().__init__(req_cols=req_cols,
                         metric_name=metric_name,
                         reference_class=reference_class,
                         needs_pos_and_neg=needs_pos_and_neg,
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
