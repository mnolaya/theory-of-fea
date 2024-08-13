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

    # def N(self, eta_1: float, eta_2: float) -> np.ndarray:
    #     r1 = itertools.chain.from_iterable([[f(eta_1, eta_2), 0] for f in self.get_shape_funcs()])
    #     r2 = itertools.chain.from_iterable([[0, f(eta_1, eta_2)] for f in self.get_shape_funcs()])
    #     return [r1, r2]
    
    def get_shape_funcs(self) -> list[np.ndarray]:
        return [self.shape_n1, self.shape_n2, self.shape_n3, self.shape_n4]
    
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