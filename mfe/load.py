from __future__ import annotations

from attrs import define, field
import numpy as np
import numpy.typing as npt

import mfe.baseclasses
import mfe.utils

SURFACE_TRACTION_SETUP = {
    '+x': {
        'J_components':  [(1, 0, ), (0, 1, )],
        'eta_component': 0,
        'eta_val': 1,
    },
    '-x': {
        'J_components':  [(1, 0, ), (0, 1, )],
        'eta_component': 0,
        'eta_val': -1,
    },
    '+y': {
        'J_components': [(0, 0, ), (0, 1, )],
        'eta_component': 1,
        'eta_val': 1,
    },
    '-y': {
        'J_components': [(0, 0, ), (0, 1, )],
        'eta_component': 1,
        'eta_val': -1,
    },
}

def _polynomial(x: np.ndarray, constants: np.ndarray) -> np.ndarray:
    return np.sum([constants[i]*x**i for i in range(len(constants))])

@define
class SurfaceTraction:
    face: str
    constants: npt.ArrayLike
    thickness: float = 1
    order: int = field(init=False)
    funcs: tuple = field(init=False)
    _ndim: int = 2
    _setup_dict: dict = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.order = len(self.constants) - 1
        self.funcs = tuple(_polynomial for _ in range(self._ndim))
        self._setup_dict = SURFACE_TRACTION_SETUP[self.face].copy()

    def compute_J_det_surf(self, elem: mfe.baseclasses.Element2D, natural_coords: np.ndarray) -> np.ndarray:
        '''
        Compute the Jacobi-determinant along a surface subject to traction.
        '''
        # Compute the Jacobian
        dN = elem.compute_dN(natural_coords)
        J = elem.compute_J(dN)

        # Get the Jacobian components used for computation
        J1, J2 = J[:, :, *self._setup_dict['J_components'][0]], J[:, :, *self._setup_dict['J_components'][1]]

        # Compute the Jacobi-determinant along the surface
        return (J1**2 + J2**2)**0.5
    
    def compute_force_vector(self, elem: mfe.baseclasses.Element2D, natural_coords: np.ndarray) -> np.ndarray:
        '''
        Compute the force vector due to the surface traction.
        '''
        # Compute the shape functions for the grid of integration point coordinates along the surface
        N = elem.compute_N(natural_coords)
        N_transpose = np.transpose(N, axes=(0, 1, 3, 2))

        # Get the coordinates of the integration points in the local element system
        x_coords = elem.interpolate(elem.x_element, natural_coords)

        # Compute the surface traction forces at the integration points
        grid_shape = natural_coords.shape[0:2]
        f_s = np.array([f(x_coords, *self.constants) for f in self.funcs]).reshape(self._ndim, 1, *grid_shape)
        f_s = mfe.utils.shift_ndarray_for_vectorization(f_s)

        # Compute the force vector
        # f = w_ij*J_det_surf*np.matmul(N_transpose, f_surf)
        # return thickness*np.sum(f, axis=(0, 1))



    # @classmethod
    # def build_from_face_loc(cls, face: str, poly_constants: npt.ArrayLike, thickness: float = 1, ndim: int = 2) -> SurfaceTraction:
    #     c


if __name__ == '__main__':
    bc = SurfaceTraction(face='+x', constants=np.array([0, 1, 2]), thickness=1.3, ndim=2)
    print(bc)


    