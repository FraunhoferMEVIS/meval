from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt


BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
LabelArray: TypeAlias = BoolArray | IntArray
MixedLabelArray: TypeAlias = npt.NDArray[np.bool_ | np.integer[Any]]
FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
NumericArray: TypeAlias = npt.NDArray[np.bool_ | np.integer[Any] | np.floating[Any]]
GenericArray: TypeAlias = npt.NDArray[np.generic]
