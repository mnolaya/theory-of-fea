import itertools

import numpy as np
from attrs import define

from mfea.baseclasses import Element2D, Node

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
    
    def get_shape_funcs(self) -> list[np.ndarray]:
        return [
            self.shape_n1, 
            self.shape_n2, 
            self.shape_n3,
            self.shape_n4
        ]
    
    def dN(self, eta_1: float, eta_2: float) -> np.ndarray:
        r1 = [self.dN1_1(eta_2), 0, self.dN2_1(eta_2), 0, self.dN3_1(eta_2), 0, self.dN4_1(eta_2), 0]
        r2 = [self.dN1_2(eta_1), 0, self.dN2_2(eta_1), 0, self.dN3_2(eta_1), 0, self.dN4_2(eta_1), 0]
        r3 = [0, self.dN1_1(eta_2), 0, self.dN2_1(eta_2), 0, self.dN3_1(eta_2), 0, self.dN4_1(eta_2)]
        r4 = [0, self.dN1_2(eta_1), 0, self.dN2_2(eta_1), 0, self.dN3_2(eta_1), 0, self.dN4_2(eta_1)]
        return np.array([r1, r2, r3, r4])
    
    def J(self, dN: np.ndarray) -> np.ndarray:
        J_col = np.matmul(dN, self.x_element)
        return np.array(
            [
                [J_col[0], J_col[1]], 
                [J_col[2], J_col[3]]
            ]
        ), J_col
    
    def B(self, eta_1: float, eta_2: float) -> np.ndarray:
        dN = self.dN(eta_1, eta_2)
        J, J_col = self.J(dN)
        zs = np.zeros(J.shape)
        J_star = np.vstack([np.hstack([J, zs]), np.hstack([zs, J])])
        return np.matmul(A_2D, np.matmul(np.linalg.inv(J_star), dN))
    
    def strain(self, eta_1: float, eta_2: float) -> np.ndarray:
        return np.matmul(self.B(eta_1, eta_2), self.x_element)
        
    @staticmethod
    def shape_n1(eta_1: float, eta_2: float) -> float:
        return 0.25*(eta_1 - 1)*(eta_2 - 1)

    @staticmethod
    def shape_n2(eta_1: float, eta_2: float) -> float:
        return -0.25*(eta_1 + 1)*(eta_2 - 1)

    @staticmethod
    def shape_n3(eta_1: float, eta_2: float) -> float:
        return 0.25*(eta_1 + 1)*(eta_2 + 1)

    @staticmethod
    def shape_n4(eta_1: float, eta_2: float) -> float:
        return -0.25*(eta_1 - 1)*(eta_2 + 1)
    
    @staticmethod
    def dN1_1(eta_2: float) -> float:
        return 0.25*(eta_2 - 1)
    
    @staticmethod
    def dN1_2(eta_1: float) -> float:
        return 0.25*(eta_1 - 1)
    
    @staticmethod
    def dN2_1(eta_2: float) -> float:
        return -0.25*(eta_2 - 1)
    
    @staticmethod
    def dN2_2(eta_1: float) -> float:
        return -0.25*(eta_1 + 1)
    
    @staticmethod
    def dN3_1(eta_2: float) -> float:
        return 0.25*(eta_2 + 1)
    
    @staticmethod
    def dN3_2(eta_1: float) -> float:
        return 0.25*(eta_1 + 1)
    
    @staticmethod
    def dN4_1(eta_2: float) -> float:
        return -0.25*(eta_2 + 1)
    
    @staticmethod
    def dN4_2(eta_1: float) -> float:
        return -0.25*(eta_1 - 1)

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