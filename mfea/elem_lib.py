import itertools

import numpy as np
from attrs import define

from mfea.baseclasses import Element2D, Node
import mfea.utils

A_2D = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

@define
class Linear2D(Element2D):
    @classmethod
    def from_element_coords(cls, coords: list[np.ndarray]):
        nodes = [
            Node(element_id=1, element_coords=coords[0], natural_coords=np.array([-1, -1])),
            Node(element_id=2, element_coords=coords[1], natural_coords=np.array([1, -1])),
            Node(element_id=3, element_coords=coords[2], natural_coords=np.array([1, 1])),
            Node(element_id=4, element_coords=coords[3], natural_coords=np.array([-1, 1])),
        ]
        return cls(nodes)
    
    def N(self, eta: np.ndarray) -> np.ndarray:
        # Compute the value of the shape functions for the entire grid
        # and then assemble into ngrid x ngrid N matrices for the element
        N1 = self.shape_n1(eta)
        N2 = self.shape_n2(eta)
        N3 = self.shape_n3(eta)
        N4 = self.shape_n4(eta)

        return self._assemble_N([N1, N2, N3, N4])
    
    def get_shape_funcs(self) -> list[np.ndarray]:
        return [
            self.shape_n1, 
            self.shape_n2, 
            self.shape_n3,
            self.shape_n4
        ]
    
    # def dN(self, eta_1: float, eta_2: float) -> np.ndarray:
    def dN(self, eta: np.ndarray) -> np.ndarray:
        # Compute the value of shape function derivatives for the entire grid
        # with respect to the first coordinate direction
        dN1_1 = self.dN1_1(eta)
        dN2_1 = self.dN2_1(eta)
        dN3_1 = self.dN3_1(eta)
        dN4_1 = self.dN4_1(eta)

        # Compute the value of shape function derivatives for the entire grid
        # with respect to the second coordinate direction
        dN1_2 = self.dN1_2(eta)
        dN2_2 = self.dN2_2(eta)
        dN3_2 = self.dN3_2(eta)
        dN4_2 = self.dN4_2(eta)

        return self._assemble_dN(sfuncs_1=[dN1_1, dN2_1, dN3_1, dN4_1], sfuncs_2=[dN1_2, dN2_2, dN3_2, dN4_2])
    
    def J(self, dN: np.ndarray) -> np.ndarray:
        # Assemble q array using numpy broadcasting for vectorized matrix multiplication
        q = mfea.utils.to_col_vec(self.x_element)
        q = mfea.utils.broadcast_ndarray_for_vectorziation(q, dN.shape[0])

        # Compute the Jacobian matrix for the element
        J_col = np.matmul(dN, q)
        J_mat = np.array(
            [
                [J_col[:, :, 0, 0], J_col[:, :, 1, 0]], 
                [J_col[:, :, 2, 0], J_col[:, :, 3, 0]]
            ]
        )

        # Assemble the full Jacobian matrix for the element used to compute the B matrix
        return self._assemble_J(J_mat)
    
    def B(self, eta: np.ndarray) -> np.ndarray:
        # Compute...
        dN = self.dN(eta)  #  derivative matrix 
        J = self.J(dN)  # Full Jacobian

        # Assemble A matrix for mapping displacement gradients to strains in Voigt notation using numpy broadcasting for vectorized matrix multiplication
        A = mfea.utils.broadcast_ndarray_for_vectorziation(A_2D, dN.shape[0])

        # Compute B matrix for the element
        return np.matmul(A, np.matmul(np.linalg.inv(J), dN))
        
    @staticmethod
    def shape_n1(eta: np.ndarray) -> float | np.ndarray:
        return 0.25*(eta[0] - 1)*(eta[1] - 1)

    @staticmethod
    def shape_n2(eta: np.ndarray) -> float | np.ndarray:
        return -0.25*(eta[0] + 1)*(eta[1] - 1)

    @staticmethod
    def shape_n3(eta: np.ndarray) -> float | np.ndarray:
        return 0.25*(eta[0]+ 1)*(eta[1] + 1)

    @staticmethod
    def shape_n4(eta: np.ndarray) -> float | np.ndarray:
        return -0.25*(eta[0] - 1)*(eta[1] + 1)
    
    @staticmethod
    def dN1_1(eta: np.ndarray) -> float | np.ndarray:
        return 0.25*(eta[1] - 1)
    
    @staticmethod
    def dN1_2(eta: np.ndarray) -> float | np.ndarray:
        return 0.25*(eta[0] - 1)
    
    @staticmethod
    def dN2_1(eta: np.ndarray) -> float | np.ndarray:
        return -0.25*(eta[1] - 1)
    
    @staticmethod
    def dN2_2(eta: np.ndarray) -> float | np.ndarray:
        return -0.25*(eta[0] + 1)
    
    @staticmethod
    def dN3_1(eta: np.ndarray) -> float | np.ndarray:
        return 0.25*(eta[1] + 1)
    
    @staticmethod
    def dN3_2(eta: np.ndarray) -> float | np.ndarray:
        return 0.25*(eta[0] + 1)
    
    @staticmethod
    def dN4_1(eta: np.ndarray) -> float | np.ndarray:
        return -0.25*(eta[1] + 1)
    
    @staticmethod
    def dN4_2(eta: np.ndarray) -> float | np.ndarray:
        return -0.25*(eta[0] - 1)

@define
class Quadratic2D(Element2D):

    @classmethod
    def from_element_coords(cls, coords: list[np.ndarray]):
        nodes = [
            Node(element_id=1, element_coords=coords[0], natural_coords=np.array([-1, -1])),
            Node(element_id=2, element_coords=coords[1], natural_coords=np.array([0, -1])),
            Node(element_id=3, element_coords=coords[2], natural_coords=np.array([1, -1])),
            Node(element_id=4, element_coords=coords[3], natural_coords=np.array([1, 0])),
            Node(element_id=5, element_coords=coords[4], natural_coords=np.array([1, 1])),
            Node(element_id=6, element_coords=coords[5], natural_coords=np.array([0, 1])),
            Node(element_id=7, element_coords=coords[6], natural_coords=np.array([-1, 1])),
            Node(element_id=8, element_coords=coords[7], natural_coords=np.array([-1, 0])),
        ]
        return cls(nodes)
    
    def get_shape_funcs(self) -> list[np.ndarray]:
        return [
            self.shape_n1, 
            self.shape_n2, 
            self.shape_n3, 
            self.shape_n4, 
            self.shape_n5, 
            self.shape_n6, 
            self.shape_n7, 
            self.shape_n8
        ]
    
    @staticmethod
    def shape_n1(eta_1: float, eta_2: float) -> float:
        return -0.25*(eta_1 - 1)*(eta_2 - 1)*(eta_1 + eta_2 + 1)

    @staticmethod
    def shape_n2(eta_1: float, eta_2: float) -> float:
        return 0.5*(eta_1 - 1)*(eta_1 + 1)*(eta_2 - 1)

    @staticmethod
    def shape_n3(eta_1: float, eta_2: float) -> float:
        return -0.25*(eta_1 + 1)*(eta_2 - 1)*(eta_1 - eta_2 - 1)

    @staticmethod
    def shape_n4(eta_1: float, eta_2: float) -> float:
        return -0.5*(eta_1 + 1)*(eta_2 - 1)*(eta_2 + 1)

    @staticmethod
    def shape_n5(eta_1: float, eta_2: float) -> float:
        return 0.25*(eta_1 + 1)*(eta_2 + 1)*(eta_2 + eta_1 - 1)

    @staticmethod
    def shape_n6(eta_1: float, eta_2: float) -> float:
        return -0.5*(eta_1 - 1)*(eta_1 + 1)*(eta_2 + 1)

    @staticmethod
    def shape_n7(eta_1: float, eta_2: float) -> float:
        return 0.25*(eta_1 - 1)*(eta_2 + 1)*(eta_1 - eta_2 + 1)

    @staticmethod
    def shape_n8(eta_1: float, eta_2: float) -> float:
        return 0.5*(eta_1 - 1)*(eta_2 - 1)*(eta_2 + 1)