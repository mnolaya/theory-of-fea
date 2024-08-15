import itertools

import numpy as np
from attrs import define

from mfea.baseclasses import Element2D, Node

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