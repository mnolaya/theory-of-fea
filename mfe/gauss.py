from __future__ import annotations
import itertools

import numpy as np
from attrs import define, field

from mfe.utils import shift_ndarray_for_vectorization

GAUSS_POINTS = {
    1: {
        'loc': [0.0],
        'weights': [2.0]
    },
    2: {
        'loc': [-0.57735, 0.57735],
        'weights': [1.0, 1.0]
    },
    3: {
        'loc': [-0.775, 0.0, 0.775],
        'weights': [0.556, 0.889, 0.556]
    },
}

@define
class IntegrationPoints:
    natural_coords: np.ndarray
    weights: np.ndarray
    element_coords: np.ndarray = field(init=False)

    @classmethod
    def make_ip_grid(cls, npts: int, ndim: int = 2) -> IntegrationPoints:
        # Get locations and weights from the dictionary of Gauss points
        locs = np.array(GAUSS_POINTS[npts]['loc'])
        w = np.array(GAUSS_POINTS[npts]['weights'])

        # Create grid for ip locations in natural coordinate system
        ip_grid = np.array(np.meshgrid(locs, locs))

        # Create grid for weights, then multiply elements along the first axis (e.g., w_i*w_j*w_k)
        w_grid = np.prod(np.array(np.meshgrid(w, w)), axis=0, keepdims=True)

        # Reshape for matrix multiplication
        ip_grid = ip_grid.reshape((ndim, 1, npts, npts))
        w_grid = w_grid.reshape((1, 1, npts, npts))

        # Return with correct shape for numpy vectorized matrix multiplication
        return cls(
            natural_coords=shift_ndarray_for_vectorization(ip_grid),
            weights=shift_ndarray_for_vectorization(w_grid)
        )
    
    @property
    def x_natural(self) -> np.ndarray:
        return self.natural_coords