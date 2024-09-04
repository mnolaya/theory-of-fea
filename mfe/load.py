from __future__ import annotations

from attrs import define, field
import numpy as np
import numpy.typing as npt

import mfe.baseclasses
import mfe.gauss
import mfe.utils
import mfe.elem_lib

SURFACE_TRACTION_SETUP = {
    '+x': {
        'J_components':  [(1, 0, ), (1, 1, )],
        'eta_component': 0,
        'eta_val': 1,
    },
    '-x': {
        'J_components':  [(1, 0, ), (1, 1, )],
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
    return np.sum(np.array([constants[i]*x**i for i in range(len(constants))]), axis=0)

def _generate_surf_traction_itg_pts(elem: mfe.baseclasses.Element2D, face: str) -> mfe.gauss.IntegrationPoints:
    '''
    Generate integration points used to compute a surface traction integral on an element 
    '''
    # Get the components of the integration point grid to be held constant
    eta_component = SURFACE_TRACTION_SETUP[face]['eta_component']

    # Get component constant value depending on the face the traction is applied
    eta_val = SURFACE_TRACTION_SETUP[face]['eta_val']

    # Create a copy of the integration points on the element, then set the appropriate component to a constant value
    itg_pts = mfe.gauss.IntegrationPoints(elem.integration_points.natural_coords.copy(), elem.integration_points.weights.copy())
    itg_pts.natural_coords[:, :, eta_component, :] = eta_val

    npts = 2
    ndim = 2
    gauss_pt_dict = mfe.gauss.GAUSS_POINTS[npts]
    locs = gauss_pt_dict['loc']
    weights = gauss_pt_dict['weights']
    if eta_component == 1:
        pts = np.array([np.array([[l], [eta_val]]) for l in locs]).reshape((npts, 1, ndim, 1))
    else:
        pts = np.array([np.array([[eta_val], [l]]) for l in locs]).reshape((npts, 1, ndim, 1))
    weights = np.repeat(np.prod(np.array(weights), axis=0, keepdims=True), npts).reshape((1, 1, npts, 1))
    itg_pts = mfe.gauss.IntegrationPoints(natural_coords=pts, weights=mfe.utils.shift_ndarray_for_vectorization(weights))
    return itg_pts

@define
class SurfaceTraction:
    face: str
    constants: npt.ArrayLike[np.ndarray]
    integration_points: mfe.gauss.IntegrationPoints
    thickness: float = 1
    order: int = field(init=False)
    funcs: tuple = field(init=False)
    _ndim: int = 2
    _setup_dict: dict = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._ndim = len(self.constants)
        self.order = len(self.constants[0]) - 1
        self.funcs = tuple(_polynomial for _ in self.constants)
        self._setup_dict = SURFACE_TRACTION_SETUP[self.face].copy()

    @classmethod
    def generate_on_element(cls, elem: mfe.baseclasses.Element2D, face: str, constants: np.ndarray, thickness: float = 1) -> SurfaceTraction:
        '''
        Construct a SurfaceTraction instance for an element, representing a surface traction applied to an element face.
        '''
        return cls(face, constants, _generate_surf_traction_itg_pts(elem, face), thickness)

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
        J_det_surf = (J1**2 + J2**2)**0.5
        return J_det_surf.reshape((*natural_coords.shape[0:2], 1, 1))
    
    def compute_force_vector(self, elem: mfe.baseclasses.Element2D) -> np.ndarray:
        '''
        Compute the force vector due to the surface traction.
        '''
        # Compute the shape functions for the grid of integration point coordinates along the surface
        N = elem.compute_N(self.integration_points.natural_coords)
        N_transpose = np.transpose(N, axes=(0, 1, 3, 2))

        # Get the positions of the integration points in the local element coordinate system
        self.integration_points.element_coords = elem.interpolate(elem.x_element, self.integration_points.natural_coords)

        # Compute the surface traction forces at the integration points
        grid_shape = self.integration_points.natural_coords.shape[0:2]
        f_surf = np.array([
            f(self.integration_points.element_coords[:, :, i, :], c)
            for i, (f, c) in enumerate(zip(self.funcs, self.constants))
        ]).reshape((2, 1, *grid_shape))
        f_surf = mfe.utils.shift_ndarray_for_vectorization(f_surf)

        # Compute the Jacobi-determinant along the surface
        J_det_surf = self.compute_J_det_surf(elem, self.integration_points.natural_coords)

        # Compute the force vector for each integration point, then return the sum multiplied by the thickness
        w_ij = self.integration_points.weights
        f = w_ij*J_det_surf*np.matmul(N_transpose, f_surf)
        return self.thickness*np.sum(f, axis=(0, 1))

if __name__ == '__main__':
    elem = mfe.elem_lib.Linear2D.from_element_coords(
        [
            np.array([0, 0]), 
            np.array([12, -1]), 
            np.array([15, 8]), 
            np.array([-1, 10])
        ], num_pts=3
    )
    bc = SurfaceTraction.generate_on_element(elem=elem, face='+x', constants=[np.array([4, 3]), np.array([5, 1])], thickness=1.3)
    f_s = bc.compute_force_vector(elem)
    print(f_s)


    