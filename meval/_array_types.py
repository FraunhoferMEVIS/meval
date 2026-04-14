from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd


BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
BoolSeriesMask: TypeAlias = pd.Series
BoolArrayMask: TypeAlias = BoolArray
IndexArrayMask: TypeAlias = IntArray
MaskLike: TypeAlias = BoolSeriesMask | BoolArrayMask | IndexArrayMask
LabelArray: TypeAlias = BoolArray | IntArray
MixedLabelArray: TypeAlias = npt.NDArray[np.bool_ | np.integer[Any]]
FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
NumericArray: TypeAlias = npt.NDArray[np.bool_ | np.integer[Any] | np.floating[Any]]
GenericArray: TypeAlias = npt.NDArray[np.generic]
