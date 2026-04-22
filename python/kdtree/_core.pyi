from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

class KDTree:
    data: NDArray[np.float64]
    ndim: int
    n_points: int
    leafsize: int

    def __init__(self, data: ArrayLike, *, leafsize: int = 32, copy_data: bool = False) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def query(
        self,
        x: ArrayLike,
        *,
        k: int = 1,
        p: float = 2.0,
        max_distance: float | None = None,
        eps: float = 0.0,
        parallel: bool | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
