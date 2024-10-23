import itertools
from typing import Callable
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from attrs import define, field

import mfe.utils
from mfe.gauss import IntegrationPoints

DEBUG_TOL = 1e-6
A_2D = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

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
    integration_points: IntegrationPoints
    nnodes: int = field(init=False)
    D: np.ndarray = field(init=False)
    thickness: float = 1.0
    _debug: bool = False
    _ndim: int = 2

    def __attrs_post_init__(self):
        self.nnodes = len(self.nodes)
    
    @abstractmethod
    def get_shape_funcs(self) -> list:
        ...
    
    @abstractmethod
    def get_shape_func_derivatives(self) -> list:
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

    def compute_J(self, dN: np.ndarray) -> np.ndarray:
        # Assemble q array using numpy broadcasting for vectorized matrix multiplication
        q = mfe.utils.to_col_vec(self.x_element)
        # q = mfe.utils.broadcast_ndarray_for_vectorziation(q, dN.shape[0:2])

        # Compute the Jacobian matrix for the element
        J_col = np.matmul(dN, q)
        J_mat = np.array(
            [
                [J_col[:, :, 0, 0], J_col[:, :, 2, 0]], 
                [J_col[:, :, 1, 0], J_col[:, :, 3, 0]]
            ]
        )

        # Assemble the full Jacobian matrix for the element used to compute the B matrix
        return self._assemble_J(J_mat)
    
    def compute_B(self, dN: np.ndarray, J: np.ndarray) -> np.ndarray:
        # Assemble A matrix for mapping displacement gradients to strains in Voigt notation using numpy broadcasting for vectorized matrix multiplication
        # A = mfe.utils.broadcast_ndarray_for_vectorziation(A_2D, dN.shape[0:2])

        # Compute B matrix for the element
        return np.matmul(A_2D, np.matmul(np.linalg.inv(J), dN))
    
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
    
    def map_to_element(self, nodal_vec: np.ndarray, natural_grid: np.ndarray) -> np.ndarray:
        # Compute N for the array of coordinates
        N = self.compute_N(natural_grid)

        # Assemble q array using numpy broadcasting for vectorized matrix multiplication
        q = mfe.utils.to_col_vec(nodal_vec)

        # Map quantity from nodes to element using shape functions: phi = N*q 
        # where q is a vector of known quantities at the element nodes
        # and phi is the value of the quantity within the element
        return np.matmul(N, q)
    
    # def compute_strain(self, nodal_disp: np.ndarray, natural_grid: np.ndarray | None = None) -> np.ndarray:
    #     '''
    #     Compute strains given nodal displacements and an array of coordinates 
    #     in the natural coordinate system.
    #     '''
    #     if natural_grid is None: natural_grid = self.integration_points.natural_coords
    #     # Compute B-matrix...
    #     dN = self.compute_dN(natural_grid)  # Shape function derivative matrix
    #     J = self.compute_J(dN)  # Full Jacobian matrix
    #     B = self.compute_B(dN, J)  # B-matrix for computing strains from nodal displacements 

    #     # Compute strains
    #     q = mfe.utils.to_col_vec(nodal_disp)
    #     # q = mfe.utils.broadcast_ndarray_for_vectorziation(q, natural_grid.shape[0:2])
    #     return np.matmul(B, q)
    
    def compute_stress(self, eps: np.ndarray, D: np.ndarray) -> np.ndarray:
        '''
        Compute stresses from strains computed for the element.
        sigma = [D][eps]
        '''
        return np.matmul(D, eps)
    
    def compute_strain_energy_density(self, sigma: np.ndarray, eps: np.ndarray) -> np.ndarray:
        '''
        Compute the strain energy for the element.
        psi = 1/2*[sigma]'[eps]
        '''
        return 0.5*np.matmul(np.transpose(sigma, axes=(0, 1, 3, 2)), eps)
    
    def _compute_k(self, D: np.ndarray, thickness: float = 1) -> np.ndarray:
        '''
        Compute the stiffness matrix (k) for the element.
        psi = 1/2*[sigma]'[eps]
        '''        
        # Compute for the integration points...
        dN = self.compute_dN(self.integration_points.natural_coords)
        J = self.compute_J(dN)
        B = self.compute_B(dN, J)

        # Get the Jacobi-determinant for each of the grid points
        J_det = np.linalg.det(J[:, :, 0:2, 0:2]).reshape(
            (*self.integration_points.natural_coords.shape[0:2], 1, 1)
        )

        # Transpose B for matrix multiplication
        B_transpose = np.transpose(B, axes=(0, 1, 3, 2))

        # Compute stiffness matrix for each integration point, then return sum multiplied by the thickness
        w_ij = self.integration_points.weights
        k = w_ij*np.matmul(B_transpose, np.matmul(D, B))*J_det
        return thickness*np.sum(k, axis=(0, 1))
    
    def compute_k(self) -> np.ndarray:
        '''
        Compute the stiffness matrix (k) for the element.
        psi = 1/2*[sigma]'[eps]
        '''        
        # Compute for the integration points...
        dN = self.compute_dN(self.integration_points.natural_coords)
        J = self.compute_J(dN)
        B = self.compute_B(dN, J)

        # Get the Jacobi-determinant for each of the grid points
        J_det = np.linalg.det(J[:, :, 0:2, 0:2]).reshape(
            (*self.integration_points.natural_coords.shape[0:2], 1, 1)
        )

        # Transpose B for matrix multiplication
        B_transpose = np.transpose(B, axes=(0, 1, 3, 2))

        # Compute stiffness matrix for each integration point, then return sum multiplied by the thickness
        w_ij = self.integration_points.weights
        k = w_ij*np.matmul(B_transpose, np.matmul(self.D, B))*J_det
        return self.thickness*np.sum(k, axis=(0, 1))
    
    def compute_strain(self, B: np.ndarray, q: np.ndarray) -> np.ndarray:
        '''
        Compute the strains for the element.
        '''
        # Convert q to column vector if not already in correct form
        q = mfe.utils.to_col_vec(q)

        # Check for vectorization
        if len(B.shape) == 4:
            if B.shape[2] != 3 or B.shape[3] != 2*self.nnodes: 
                print(f'error: B matrix is of shape {B.shape}. for vectorization, it must be of shape (n, m, 3, 2*nnodes)')
                exit()
            q = mfe.utils.broadcast_ndarray_for_vectorziation(q, B.shape[0:2])
        elif B.shape != (4, 2*self.nnodes):
            print(f'error: B matrix is of shape {B.shape}, but must be of shape (3, 2*nnodes)')
        return np.matmul(B, q)
    
    def compute_stress(self, D: np.ndarray, eps: np.ndarray) -> np.ndarray:
        '''
        Compute the stresses for the element.
        '''
        # Check for vectorization
        if len(eps.shape) == 4:
            if eps.shape[2] != 3 or eps.shape[3] != 1: 
                print(f'error: strain matrix is of shape {eps.shape}. for vectorization, it must be of shape (n, m, 3, 1)')
                exit()
            D = mfe.utils.broadcast_ndarray_for_vectorziation(D, eps.shape[0:2])
        elif eps.shape != (3, 1):
            print(f'error: strain matrix is of shape {eps.shape}, but must be of shape (3, 1)')
        return np.matmul(D, eps)


