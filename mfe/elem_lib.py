from __future__ import annotations
import itertools
import enum

import numpy as np
import numpy.typing as npt
from attrs import define

from mfe.gauss import IntegrationPoints
from mfe.baseclasses import Element2D, Node
import mfe.utils

def _set_elem_coords(nodes: list[Node]) -> None:
    elem_origin = nodes[0].global_coords
    for node in nodes:
        node.element_coords = node.global_coords - elem_origin

@define
class Linear2D(Element2D):

    @classmethod
    def from_element_coords(cls, coords: list[np.ndarray], num_pts: int = 2) -> Linear2D:
        nodes = [
            Node(element_id=1, global_coords=coords[0], natural_coords=np.array([-1, -1])),
            Node(element_id=2, global_coords=coords[1], natural_coords=np.array([1, -1])),
            Node(element_id=3, global_coords=coords[2], natural_coords=np.array([1, 1])),
            Node(element_id=4, global_coords=coords[3], natural_coords=np.array([-1, 1])),
        ]
        _set_elem_coords(nodes)
        itg_pts = IntegrationPoints.make_ip_grid(num_pts, ndim=2)
        return cls(nodes, integration_points=itg_pts)
    
    def get_shape_funcs(self) -> list[np.ndarray]:
        return [
            self.shape_n1, 
            self.shape_n2, 
            self.shape_n3,
            self.shape_n4
        ]
    
    def get_shape_func_derivatives(self) -> tuple[list[np.ndarray]]:
        return [
            self.dN1_1, self.dN2_1, self.dN3_1, self.dN4_1
        ], [
            self.dN1_2, self.dN2_2, self.dN3_2, self.dN4_2
        ]
    
    def compute_N(self, natural_grid: np.ndarray) -> np.ndarray:
        # Compute the value of the shape functions for the entire grid
        # and then assemble into ngrid x ngrid N matrices for the element
        N1 = self.shape_n1(natural_grid)
        N2 = self.shape_n2(natural_grid)
        N3 = self.shape_n3(natural_grid)
        N4 = self.shape_n4(natural_grid)

        return self._assemble_N([N1, N2, N3, N4])

    def compute_dN(self, natural_grid: np.ndarray) -> np.ndarray:
        # Compute the value of shape function derivatives for the entire grid
        # with respect to the first coordinate direction
        dN1_1 = self.dN1_1(natural_grid)
        dN2_1 = self.dN2_1(natural_grid)
        dN3_1 = self.dN3_1(natural_grid)
        dN4_1 = self.dN4_1(natural_grid)

        # Compute the value of shape function derivatives for the entire grid
        # with respect to the second coordinate direction
        dN1_2 = self.dN1_2(natural_grid)
        dN2_2 = self.dN2_2(natural_grid)
        dN3_2 = self.dN3_2(natural_grid)
        dN4_2 = self.dN4_2(natural_grid)

        return self._assemble_dN(sfuncs_1=[dN1_1, dN2_1, dN3_1, dN4_1], sfuncs_2=[dN1_2, dN2_2, dN3_2, dN4_2])
        
    @staticmethod
    def shape_n1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return (1/4)*(eta_1 - 1)*(eta_2 - 1)

    @staticmethod
    def shape_n2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_1 + 1)*(eta_2 - 1)

    @staticmethod
    def shape_n3(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_1 + 1)*(eta_2 + 1)

    @staticmethod
    def shape_n4(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_1 - 1)*(eta_2 + 1)
    
    @staticmethod
    def dN1_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_2 - 1)
    
    @staticmethod
    def dN1_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_1 - 1)
    
    @staticmethod
    def dN2_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_2 - 1)
    
    @staticmethod
    def dN2_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_1 + 1)
    
    @staticmethod
    def dN3_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_2 + 1)
    
    @staticmethod
    def dN3_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_1 + 1)
    
    @staticmethod
    def dN4_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_2 + 1)
    
    @staticmethod
    def dN4_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_1 - 1)

@define
class Quadratic2D(Element2D):

    @classmethod
    def from_element_coords(cls, coords: list[np.ndarray], num_pts: int = 3) -> Quadratic2D:
        nodes = [
            Node(element_id=1, global_coords=coords[0], natural_coords=np.array([-1, -1])),
            Node(element_id=2, global_coords=coords[1], natural_coords=np.array([0, -1])),
            Node(element_id=3, global_coords=coords[2], natural_coords=np.array([1, -1])),
            Node(element_id=4, global_coords=coords[3], natural_coords=np.array([1, 0])),
            Node(element_id=5, global_coords=coords[4], natural_coords=np.array([1, 1])),
            Node(element_id=6, global_coords=coords[5], natural_coords=np.array([0, 1])),
            Node(element_id=7, global_coords=coords[6], natural_coords=np.array([-1, 1])),
            Node(element_id=8, global_coords=coords[7], natural_coords=np.array([-1, 0])),
        ]
        _set_elem_coords(nodes)
        # if num_pts < 3: print('Warning!')
        itg_pts = IntegrationPoints.make_ip_grid(num_pts, ndim=2)
        return cls(nodes, integration_points=itg_pts)
    
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
    
    def get_shape_func_derivatives(self) -> tuple[list[np.ndarray]]:
        return [
            self.dN1_1, self.dN2_1, self.dN3_1, self.dN4_1, self.dN5_1, self.dN6_1, self.dN7_1, self.dN8_1
        ], [
            self.dN1_2, self.dN2_2, self.dN3_2, self.dN4_2, self.dN5_2, self.dN6_2, self.dN7_2, self.dN8_2
        ]
    
    def compute_N(self, natural_grid: np.ndarray) -> np.ndarray:
        # Compute the value of the shape functions for the entire grid
        # and then assemble into ngrid x ngrid N matrices for the element
        N1 = self.shape_n1(natural_grid)
        N2 = self.shape_n2(natural_grid)
        N3 = self.shape_n3(natural_grid)
        N4 = self.shape_n4(natural_grid)
        N5 = self.shape_n5(natural_grid)
        N6 = self.shape_n6(natural_grid)
        N7 = self.shape_n7(natural_grid)
        N8 = self.shape_n8(natural_grid)

        return self._assemble_N([N1, N2, N3, N4, N5, N6, N7, N8])
    
    def compute_dN(self, natural_grid: np.ndarray) -> np.ndarray:
        # Compute the value of shape function derivatives for the entire grid
        # with respect to the first coordinate direction
        dN1_1 = self.dN1_1(natural_grid)
        dN2_1 = self.dN2_1(natural_grid)
        dN3_1 = self.dN3_1(natural_grid)
        dN4_1 = self.dN4_1(natural_grid)
        dN5_1 = self.dN5_1(natural_grid)
        dN6_1 = self.dN6_1(natural_grid)
        dN7_1 = self.dN7_1(natural_grid)
        dN8_1 = self.dN8_1(natural_grid)

        # Compute the value of shape function derivatives for the entire grid
        # with respect to the second coordinate direction
        dN1_2 = self.dN1_2(natural_grid)
        dN2_2 = self.dN2_2(natural_grid)
        dN3_2 = self.dN3_2(natural_grid)
        dN4_2 = self.dN4_2(natural_grid)
        dN5_2 = self.dN5_2(natural_grid)
        dN6_2 = self.dN6_2(natural_grid)
        dN7_2 = self.dN7_2(natural_grid)
        dN8_2 = self.dN8_2(natural_grid)

        return self._assemble_dN(
            sfuncs_1=[dN1_1, dN2_1, dN3_1, dN4_1, dN5_1, dN6_1, dN7_1, dN8_1], 
            sfuncs_2=[dN1_2, dN2_2, dN3_2, dN4_2, dN5_2, dN6_2, dN7_2, dN8_2]
        )
    
    @staticmethod
    def shape_n1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_1 - 1)*(eta_2 - 1)*(eta_1 + eta_2 + 1)

    @staticmethod
    def shape_n2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.5*(eta_1 - 1)*(eta_1 + 1)*(eta_2 - 1)

    @staticmethod
    def shape_n3(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.25*(eta_1 + 1)*(eta_2 - 1)*(eta_1 - eta_2 - 1)

    @staticmethod
    def shape_n4(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.5*(eta_1 + 1)*(eta_2 - 1)*(eta_2 + 1)

    @staticmethod
    def shape_n5(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_1 + 1)*(eta_2 + 1)*(eta_2 + eta_1 - 1)

    @staticmethod
    def shape_n6(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return -0.5*(eta_1 - 1)*(eta_1 + 1)*(eta_2 + 1)

    @staticmethod
    def shape_n7(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.25*(eta_1 - 1)*(eta_2 + 1)*(eta_1 - eta_2 + 1)

    @staticmethod
    def shape_n8(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        return 0.5*(eta_1 - 1)*(eta_2 - 1)*(eta_2 + 1)
    
    @staticmethod
    def dN1_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -(0.25*eta_1 - 0.25)*(eta_2 - 1) - 0.25*(eta_2 - 1)*(eta_1 + eta_2 + 1)
        return (1/4)*(-(eta_1 - 1)*(eta_2 - 1) - (eta_2 - 1)*(eta_1 + eta_2 + 1))
    
    @staticmethod
    def dN1_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -((1/4)*eta_1 - (1/4))*(eta_2 - 1) - ((1/4)*eta_1 - (1/4))*(eta_1 + eta_2 + 1)
        return (1/4)*(-(eta_1 - 1)*(eta_2 - 1) - (eta_1 - 1)*(eta_1 + eta_2 + 1))
    
    @staticmethod
    def dN2_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (0.5*eta_1 - 0.5)*(eta_2 - 1) + 0.5*(eta_1 + 1)*(eta_2 - 1)
        return (1/2)*((eta_1 - 1)*(eta_2 - 1) + (eta_1 + 1)*(eta_2 -1))
    
    @staticmethod
    def dN2_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return ((1/2)*eta_1 - (1/2))*(eta_1 + 1)
        return (1/2)*(eta_1 - 1)*(eta_1 + 1)
    
    @staticmethod
    def dN3_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -((1/4)*eta_1 + (1/4))*(eta_2 - 1) + (1/4)*(eta_2 - 1)*(eta_2 - eta_1 + 1)
        return (1/4)*(-(eta_1 + 1)*(eta_2 - 1) + (eta_2 - 1)*(eta_2 - eta_1 + 1))
    
    @staticmethod
    def dN3_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return ((1/4)*eta_1 + (1/4))*(eta_2 - 1) + ((1/4)*eta_1 + (1/4))*(eta_2 - eta_1 + 1)
        return (1/4)*((eta_1 + 1)*(eta_2 - 1) + (eta_1 + 1)*(eta_2 - eta_1 + 1))
    
    @staticmethod
    def dN4_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -(1/2)*(eta_2 - 1)*(eta_2 + 1)
        return -(1/2)*(eta_2 - 1)*(eta_2 + 1)

    @staticmethod
    def dN4_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -((1/2)*eta_1 + (1/2))*(eta_2 - 1) - ((1/2)*eta_1 + (1/2))*(eta_2 + 1)
        return (1/2)*(-(eta_1 + 1)*(eta_2 + 1) - (eta_1 + 1)*(eta_2 - 1))
    
    @staticmethod
    def dN5_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (1/4)*(eta_1 + 1)*(eta_2 + 1) - (1/4)*(eta_2 + 1)*(eta_1 + eta_2 - 1)
        return (1/4)*((eta_1 + 1)*(eta_2 + 1) + (eta_2 + 1)*(eta_1 + eta_2 - 1))
    
    @staticmethod
    def dN5_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (1/4)*(eta_1 + 1)*(eta_2 + 1) + (1/4)*(eta_1 + 1)*(eta_1 + eta_2 - 1)
        return (1/4)*((eta_1 + 1)*(eta_2 + 1) + (eta_1 + 1)*(eta_1 + eta_2 - 1))
    
    @staticmethod
    def dN6_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -(1/4)*(eta_1 - 1)*(eta_2 + 1) - (1/2)*(eta_1 + 1)*(eta_2 + 1)
        return (1/2)*(-(eta_1 - 1)*(eta_2 + 1) - (eta_1 + 1)*(eta_2 + 1))
    
    @staticmethod
    def dN6_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return -(1/2)*(eta_1 - 1)*(eta_1 + 1)
        return (1/2)*(-(eta_1 - 1)*(eta_1 + 1))
    
    @staticmethod
    def dN7_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (1/4)*(eta_1 - 1)*(eta_2 + 1) + (1/4)*(eta_2 + 1)*(eta_1 - eta_2 + 1)
        return (1/4)*((eta_1 - 1)*(eta_2 + 1) + (eta_2 + 1)*(eta_1 - eta_2 + 1))
    
    @staticmethod
    def dN7_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (1/4)*(eta_1 - 1)*(eta_2 + 1) + (1/4)*(eta_1 - 1)*(eta_1 - eta_2 + 1)
        return (1/4)*(-(eta_1 - 1)*(eta_2 + 1) + (eta_1 - 1)*(eta_1 - eta_2 + 1))
    
    @staticmethod
    def dN8_1(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (1/2)*(eta_2 - 1)*(eta_2 + 1)
        return (1/2)*(eta_2 - 1)*(eta_2 + 1)
    
    @staticmethod
    def dN8_2(eta: np.ndarray) -> float | np.ndarray:
        eta_1, eta_2 = mfe.utils.components_from_grid(eta)
        # return (1/2)*(eta_1 - 1)*(eta_2 + 1) + (1/2)*(eta_1 - 1)*(eta_2 - 1)
        # return (1/2)*((eta_1 - 1)*(eta_2 + 1) + (eta_1 - 1)*(eta_2 - 1))
        return (1/2)*((eta_1 - 1)*(eta_2 + 1) + (eta_1 - 1)*(eta_2 - 1))