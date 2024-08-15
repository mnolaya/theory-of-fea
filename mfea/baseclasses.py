import itertools
from typing import Callable
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from attrs import define, field

from mfea.utils import make_natural_grid

DEBUG_TOL = 1e-6

@define
class Node:
    element_id: int
    element_coords: np.ndarray
    natural_coords: np.ndarray

@define
class Element2D:
    nodes: list[Node]
    nnodes: int = field(init=False)
    _debug: bool = False

    def __attrs_post_init__(self):
        self.nnodes = len(self.nodes)
    
    @abstractmethod
    def get_shape_funcs(self) -> list:
        ...

    def N(self, eta_1: float, eta_2: float) -> np.ndarray:
        r1 = itertools.chain.from_iterable([[f(eta_1, eta_2), 0] for f in self.get_shape_funcs()])
        r2 = itertools.chain.from_iterable([[0, f(eta_1, eta_2)] for f in self.get_shape_funcs()])
        return np.array([list(r1), list(r2)])
    
    # @abstractmethod
    def J(self, eta_1: float, eta_2: float) -> np.ndarray:
        ...

    @property
    def x_natural(self) -> np.ndarray:
        return list(itertools.chain.from_iterable([node.natural_coords for node in self.nodes]))

    @property
    def x_element(self) -> np.ndarray:
        return list(itertools.chain.from_iterable([node.element_coords for node in self.nodes]))
    
    def interpolate(self, nodal_vec: np.ndarray, nvals: int = 100) -> tuple[np.ndarray]:
        # Create grid for the shape functions
        eta_1, eta_2 = make_natural_grid(nvals)

        # Interpolate for each point on the grid
        interpolated = []
        for i in range(nvals):
            row = []
            for j in range(nvals):
                e1, e2 = eta_1[i, j], eta_2[i, j]
                N = self.N(e1, e2)
                if self._debug:
                    if sum(N[0, :]) < (1 - DEBUG_TOL):
                        print('Warning! The sum of the nodal shape functions does not equal 1!')
                        print(f'Row 1 of N: {N[0, :]}')
                        print(f'Sum: {sum(N[0, :])}')
                row.append(np.matmul(N, nodal_vec))
            interpolated.append(np.array(row))
        interpolated = np.array(interpolated)
        return interpolated[:, :, 0], interpolated[:, :, 1]