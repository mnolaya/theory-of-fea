import numpy as np
import matplotlib.pyplot as plt

import mfe

ERR_TOL = 1e-6

def test_2D_shape_functions(elem: mfe.baseclasses.Element2D, ngrid: int = 50) -> None:
    # Create a grid of coordinates in the natural coordinate system
    natural_grid = mfe.utils.make_natural_grid(ngrid)  

    # Compute shape function matrix N and shape functions individually
    N = elem.compute_N(natural_grid)

    # Verify shape
    nnodes = len(elem.nodes)
    if N.shape[-2:] != (2, 2*nnodes): 
        print(f'error: shape of N matrix for 2D element is not (2, {nnodes*2})')
        exit()

    # Verify sum of shape functions equal to 1
    if not np.all(np.sum(N, axis=3) >= 1 - ERR_TOL): print(f'error: sum of shape functions not equal to 1')

    # Verify non-shape function terms equal to 0
    if not np.all([np.all(N[:, :, 0, i - 1] == 0) for i in np.arange(nnodes, step=2)]): 
        print(f'error: shape function matrix N has too many nonzero values')
        exit()

    # Verify components of N vs. individually calculated values
    N_i = [sf(natural_grid) for sf in elem.get_shape_funcs()]
    for i, sf_vals in enumerate(N_i):
        if not np.all(N[:, :, 0, i*2] ==  sf_vals):
            print(f'error: shape function {i} is not equal to the [0, {i}] component of matrix N')
            exit()
    print('shape function tests successfully passed!')
    return

def test_2D_element(
        elem: mfe.baseclasses.Element2D,
        D: np.ndarray,
        nodal_vec: np.ndarray,
        ngrid: int = 50,
    ) -> dict[str, dict]:
    # Create a grid of coordinates in the natural coordinate system
    natural_grid = mfe.utils.make_natural_grid(ngrid)

    # Interpolate to get the grid in terms of the local element coordinate system
    x_grid = elem.interpolate(elem.x_element, natural_grid)

    # Compute shape function matrix N
    N = elem.compute_N(natural_grid)

    # Compute displacements
    u = elem.interpolate(elem.x_element, natural_grid)

    # Compute shape function derivatives and Jacobian
    dN = elem.compute_dN(natural_grid)
    J = elem.compute_J(dN)

    # Compute the B-matrix
    B = elem.compute_B(natural_grid)

    # Compute the strains
    q = mfe.utils.to_col_vec(nodal_vec)
    q = mfe.utils.broadcast_ndarray_for_vectorziation(q, ngrid)
    eps = np.matmul(B, q)

    # Compute the stress
    D = mfe.utils.broadcast_ndarray_for_vectorziation(D, ngrid)
    sigma = np.matmul(D, eps)

    # Plot all results
    figs = {
        'N': {
            'natural': mfe.plot.plot_element_shape_functions(N, natural_grid, method='scatter', coord_sys='natural'),
            'element': mfe.plot.plot_element_shape_functions(N, x_grid, method='scatter', coord_sys='element')
        },
        'J': {
            'natural': mfe.plot.plot_element_Jacobian(J, natural_grid, method='scatter', coord_sys='natural'),
            'element': mfe.plot.plot_element_Jacobian(J, x_grid, method='scatter', coord_sys='element')
        },
        'sigeps': {
            'natural': mfe.plot.plot_element_stress_strain(sigma, eps, natural_grid, method='scatter', coord_sys='natural'),
            'element': mfe.plot.plot_element_stress_strain(sigma, eps, x_grid, method='scatter', coord_sys='element')
        },
    }
    return figs       

if __name__ == '__main__':
    elem = mfe.elem_lib.Linear2D.from_element_coords(
        [
            np.array([0, 0]), 
            np.array([12, -1]), 
            np.array([15, 8]), 
            np.array([-1, 10])
        ]
    )
    mat = mfe.baseclasses.Material(E=2500, nu=0.35)
    D = mat.D_isotropic_plane_stress()
    nodal_disp = np.array([0, 0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])
    # test_2D_element(elem, D, nodal_disp)
    test_2D_shape_functions(elem)