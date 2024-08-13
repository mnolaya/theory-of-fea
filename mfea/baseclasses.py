import itertools
from typing import Callable
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from attrs import define

@define
class Node:
    element_id: int
    element_coords: np.ndarray
    natural_coords: np.ndarray

# Compute the value of a shape function for a series of grid points in the natural coordinate system.
def _compute_shape_func(shape_func: Callable, eta_grid: tuple[np.ndarray]):
    val = []
    for i in range(eta_grid[0].shape[0]):
        row = []
        for j in range(eta_grid[1].shape[0]):
            e1, e2 = eta_grid[0][i, j], eta_grid[1][i][j]
            row.append(shape_func(e1, e2))
        val.append(np.array(row))
    return np.array(val)

@define
class Element2D:
    nodes: list[Node]
    
    @abstractmethod
    def get_shape_funcs(self) -> list:
        ...

    def N(self, eta_1: float, eta_2: float) -> np.ndarray:
        r1 = itertools.chain.from_iterable([[f(eta_1, eta_2), 0] for f in self.get_shape_funcs()])
        r2 = itertools.chain.from_iterable([[0, f(eta_1, eta_2)] for f in self.get_shape_funcs()])
        return np.array([list(r1), list(r2)])

    @property
    def x_natural(self) -> np.ndarray:
        return list(itertools.chain.from_iterable([node.natural_coords for node in self.nodes]))

    @property
    def x_element(self) -> np.ndarray:
        return list(itertools.chain.from_iterable([node.element_coords for node in self.nodes]))
    
    def interpolate(self, nodal_vec: np.ndarray, nvals: int = 100) -> tuple[np.ndarray]:
        # Create grid for the shape functions
        eta_1 = np.linspace(-1, 1, nvals)
        eta_2 = np.linspace(-1, 1, nvals)
        eta_1, eta_2 = np.meshgrid(eta_1, eta_2)

        # Interpolate for each point on the grid
        interpolated = []
        for i in range(nvals):
            row = []
            for j in range(nvals):
                e1, e2 = eta_1[i, j], eta_2[i, j]
                row.append(np.matmul(self.N(e1, e2), nodal_vec))
            interpolated.append(np.array(row))
        interpolated = np.array(interpolated)
        return interpolated[:, :, 0], interpolated[:, :, 1]