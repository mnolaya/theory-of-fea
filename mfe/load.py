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

def _build_polynomial(x: np.ndarray, constants: np.ndarray) -> np.ndarray:
    return np.sum(np.array([constants[i]*x**i for i in range(len(constants))]), axis=0)

def _generate_surf_traction_itg_pts(elem: mfe.baseclasses.Element2D, face: str, order: int) -> mfe.gauss.IntegrationPoints:
    '''
    Generate integration points used to compute a surface traction integral on an element 
    '''
    # Get the components of the integration point grid to be held constant
    eta_component = SURFACE_TRACTION_SETUP[face]['eta_component']

    # Get component constant value depending on the face the traction is applied
    eta_val = SURFACE_TRACTION_SETUP[face]['eta_val']

    # Create a copy of the integration points on the element, then set the appropriate component to a constant value
    min_pts = (2*order - 1)
    itg_pts = mfe.gauss.IntegrationPoints.make_ip_grid(min_pts)
    # nat_coords = itg_pts.natural_coords.copy()
    nat_coords = elem.integration_points.natural_coords.copy()
    nat_coords[:, :, eta_component, :] = eta_val

    # Retain only the unique integration point coordinates (i.e., remove duplicate coordinates when one coordinate is held constant)
    npts = nat_coords.shape[0]
    nat_coords, _ = np.unique(nat_coords, axis=1, return_index=True)
    w = np.array(mfe.gauss.GAUSS_POINTS[npts]['weights']).reshape((npts, 1, 1, 1))

    return mfe.gauss.IntegrationPoints(
        nat_coords,
        w,
    )

@define
class SurfaceTraction:
    face: str
    constants: npt.ArrayLike[np.ndarray]
    integration_points: mfe.gauss.IntegrationPoints
    thickness: float = 1
    order: int = field(init=False)
    # funcs: tuple = field(init=False)
    _ndim: int = 2
    _setup_dict: dict = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._ndim = len(self.constants)
        # self.funcs = tuple(_build_polynomial for _ in self.constants)
        self._setup_dict = SURFACE_TRACTION_SETUP[self.face].copy()

    @classmethod
    def generate(cls, elem: mfe.baseclasses.Element2D, face: str, constants: np.ndarray, thickness: float = 1) -> SurfaceTraction:
        '''
        Construct a SurfaceTraction instance for an element, representing a surface traction applied to an element face.
        '''
        # order = len(constants[0]) - 1
        order = 1
        return cls(face, constants, _generate_surf_traction_itg_pts(elem, face, order), thickness, order)

    # def compute_fs(self, elem_coords: np.ndarray) -> np.ndarray:
    #     '''
    #     Compute the surface traction forces at the integration points in the local element coordinate system.
    #     '''
    #     grid_shape = elem_coords.shape[0:2]
    #     f_surf = np.array([
    #         f(elem_coords[:, :, i, :], c)
    #         for i, (f, c) in enumerate(zip(self.funcs, self.constants))
    #     ]).reshape((2, 1, *grid_shape))
    #     return mfe.utils.shift_ndarray_for_vectorization(f_surf)

    def compute_fs(self, elem_coords: np.ndarray) -> np.ndarray:
        '''
        Compute the surface traction forces at the integration points in the local element coordinate system.
        '''
        grid_shape = elem_coords.shape[0:2]
        f_surf = []
        for _, cgroup in enumerate(self.constants):
            if type(cgroup) != list:
                soln = np.sum(np.array([cgroup[i]*elem_coords[:, :, 0, :]**i for i in range(len(cgroup))]), axis=0)
            elif len(cgroup) == 2:
                soln = np.sum(np.array([cgroup[0][i]*elem_coords[:, :, 0, :]**i for i in range(len(cgroup[0]))]), axis=0)
                soln += np.sum(np.array([cgroup[1][i]*elem_coords[:, :, 1, :]**i for i in range(len(cgroup[1]))]), axis=0)
            f_surf.append(soln)
        f_surf = np.array(f_surf).reshape((2, 1, *grid_shape))
        return mfe.utils.shift_ndarray_for_vectorization(f_surf)

    def compute_J_det_surf(self, J: np.ndarray, grid_shape: np.ndarray) -> np.ndarray:
        '''
        Compute the Jacobi-determinant along a surface subject to traction.
        '''
        # # Compute the Jacobian
        # dN = elem.compute_dN(natural_coords)
        # J = elem.compute_J(dN)

        # Get the Jacobian components used for computation
        J1, J2 = J[:, :, *self._setup_dict['J_components'][0]], J[:, :, *self._setup_dict['J_components'][1]]

        # Compute the Jacobi-determinant along the surface
        J_det_surf = (J1**2 + J2**2)**0.5
        return J_det_surf.reshape((*grid_shape[0:2], 1, 1))
    
    def compute_force_vector(self, elem: mfe.baseclasses.Element2D) -> np.ndarray:
        '''
        Compute the force vector due to the surface traction.
        '''
        grid_shape = self.integration_points.natural_coords.shape

        # Compute the shape functions for the grid of integration points along the surface
        N = elem.compute_N(self.integration_points.natural_coords)
        N_transpose = np.transpose(N, axes=(0, 1, 3, 2))

        # Get the positions of the integration points in the local element coordinate system
        self.integration_points.element_coords = elem.map_to_element(elem.x_element, self.integration_points.natural_coords)

        # Compute the surface traction forces at the integration points
        f_surf = self.compute_fs(self.integration_points.element_coords)
        # grid_shape = self.integration_points.natural_coords.shape[0:2]
        # f_surf = np.array([
        #     f(self.integration_points.element_coords[:, :, i, :], c)
        #     for i, (f, c) in enumerate(zip(self.funcs, self.constants))
        # ]).reshape((2, 1, *grid_shape))
        # f_surf = mfe.utils.shift_ndarray_for_vectorization(f_surf)

        # Compute the Jacobi-determinant along the surface
        dN = elem.compute_dN(self.integration_points.natural_coords)
        J = elem.compute_J(dN)
        J_det_surf = self.compute_J_det_surf(J, grid_shape)

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
        ], num_pts=2
    )
    bc = SurfaceTraction.generate(elem=elem, face='+x', constants=[np.array([4, 3]), np.array([5, 1])], thickness=1.3)
    f_s = bc.compute_force_vector(elem)
    print(f_s)


    