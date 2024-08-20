import itertools
from typing import Callable
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from attrs import define, field

import mfe.utils

DEBUG_TOL = 1e-6

@define
class Material:
    E: float
    nu: float

    @property
    def G(self) -> float:
        return self.E/(2*(1 + self.nu))
    
    def D_isotropic_plane_stress(self) -> np.ndarray:
        D = np.eye(3)
        c = self.E/(1 - self.nu**2)
        D[0, 1] = self.nu
        D[1, 0] = self.nu
        D[2, 2] = (1 - self.nu)/2
        return c*D
    
    def D_isotropic_plane_strain(self) -> np.ndarray:
        D = np.eye(3)
        c = self.E/((1 + self.nu)*(1 - 2*self.nu))
        D[0, 0] = 1 - self.nu
        D[0, 1] = self.nu
        D[1, 0] = self.nu
        D[1, 1] = 1 - self.nu
        D[2, 2] = (1 - 2*self.nu)/2
        return c*D

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

    @abstractmethod
    def compute_N(self, natural_grid: np.ndarray) -> np.ndarray:
        '''
        Compute the shape function matrix for a grid of natural coordinates.
        Resulting array is of shape 2 x 2*nnodes x ngrid x ngrid, where the 2 x 2*nnodes matrix
        corresponds to a position within the element as represented by the natural grid.
        '''
        ...

    @abstractmethod
    def compute_dN(self, natural_grid: np.ndarray) -> np.ndarray:
        '''
        Compute the shape function derivative matrix for a grid of natural coordinates.
        Resulting array is of shape 2 x 2*nnodes x ngrid x ngrid, where the 2 x 2*nnodes matrix
        corresponds to a position within the element as represented by the natural grid.
        '''
        ...

    @abstractmethod
    def compute_J(self, dN: np.ndarray) -> np.ndarray:
        '''
        Compute full Jacobian matrix (e.g., np.array([[J, 0], [0, J]])) for a grid of natural element coordinates.
        Results in a stack of arrays containing the Jacobian for each grid point.
        Shape: (ngrid, ngrid, 2*nnodes, 2*nnodes) where nnodes are the number of element nodes.
        '''
        ...

    @abstractmethod
    def compute_B(self, natural_grid: np.ndarray) -> np.ndarray:
        '''
        Compute the B matrix (e.g., A*inv(J)*dN) for a grid of natural element coordinates.
        Results in a stack of arrays containing the B matrix for each grid point.
        Shape: (ngrid, ngrid, ndof, 2*nnodes) where nnodes are the number of element nodes.
        '''
        ...
    
    @staticmethod
    def _assemble_N(sfuncs: list[np.ndarray]) -> np.ndarray:
        zs = np.zeros(sfuncs[0].shape)  # Matrix of zeros for assembly
        N = np.array(
            [
                list(itertools.chain.from_iterable([[sf.copy(), zs.copy()] for sf in sfuncs])),
                list(itertools.chain.from_iterable([[zs.copy(), sf.copy()] for sf in sfuncs])),
            ]
        )
        return mfe.utils.shift_ndarray_for_vectorization(N)  # Return array with multiplication axes moved to final two positions for vectorization
    
    @staticmethod
    def _assemble_dN(sfuncs_1: list[np.ndarray], sfuncs_2: list[np.ndarray]) -> np.ndarray:
        zs = np.zeros(sfuncs_1[0].shape)  # Matrix of zeros for assembly
        dN = np.array(
            [
                list(itertools.chain.from_iterable([[sf.copy(), zs.copy()] for sf in sfuncs_1])),
                list(itertools.chain.from_iterable([[sf.copy(), zs.copy()] for sf in sfuncs_2])),
                list(itertools.chain.from_iterable([[zs.copy(), sf.copy()] for sf in sfuncs_1])),
                list(itertools.chain.from_iterable([[zs.copy(), sf.copy()] for sf in sfuncs_2])),
            ]
        )
        return mfe.utils.shift_ndarray_for_vectorization(dN)  # Return array with multiplication axes moved to final two positions for vectorization
    
    @staticmethod
    def _assemble_J(J_mat: np.ndarray) -> np.ndarray:
        zs = np.zeros(J_mat.shape)  # Matrix of zeros for assembly
        J = np.array(np.vstack([np.hstack([J_mat, zs]), np.hstack([zs, J_mat])]))
        return mfe.utils.shift_ndarray_for_vectorization(J)

    @property
    def x_natural(self) -> np.ndarray:
        return np.array(list(itertools.chain.from_iterable([node.natural_coords for node in self.nodes])))

    @property
    def x_element(self) -> np.ndarray:
        return np.array(list(itertools.chain.from_iterable([node.element_coords for node in self.nodes])))
    
    def interpolate(self, nodal_vec: np.ndarray, natural_grid: np.ndarray) -> np.ndarray:
        # Compute N for the array of coordinates
        N = self.compute_N(natural_grid)

        # Assemble q array using numpy broadcasting for vectorized matrix multiplication
        q = mfe.utils.to_col_vec(nodal_vec)
        q = mfe.utils.broadcast_ndarray_for_vectorziation(q, N.shape[0])

        # Interpolate: phi = N*q 
        # where q is a vector of known quantities at the element nodes
        # and phi is the value of the quantity within the element
        return np.matmul(N, q)