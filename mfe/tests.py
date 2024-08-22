import numpy as np
import matplotlib.pyplot as plt

import mfe.baseclasses
import mfe.elem_lib
import mfe.plot
import mfe.utils

ERR_TOL = 1e-6

def _print_test_name(test_name: str) -> None:
    print(f'*running test: {test_name}')

def _print_checking(check_name: str) -> None:
    print(f'-checking: {check_name}')

def _print_test_pass(test_name: str) -> None:
    print(f'**test passed: {test_name}\n')

def _print_err_msg(check: bool, err_msg: str) -> None:
    if check: return
    print(f'-error: {err_msg}')
    exit()

def test_2D_shape_functions(elem: mfe.baseclasses.Element2D, ngrid: int = 50) -> None:
    test_name = '2D shape function (N) verification'
    _print_test_name(test_name)

    # Create a grid of coordinates in the natural coordinate system
    natural_grid = mfe.utils.make_natural_grid(ngrid)  

    # Compute shape function matrix N and shape functions individually
    N = elem.compute_N(natural_grid)

    # Verify shape
    check_name = 'shape of N matrix'
    _print_checking(check_name)
    nnodes = len(elem.nodes)
    check = N.shape[-2:] == (2, 2*nnodes)
    _print_err_msg(check, err_msg=f'N matrix for 2D element is not (2, {nnodes*2})')

    # Verify sum of shape functions equal to 1
    check_name = 'sum of shape functions'
    _print_checking(check_name)
    check = np.all(np.sum(N, axis=3) >= 1 - ERR_TOL)
    _print_err_msg(check, err_msg=f'sum of shape functions not equal to 1')

    # Verify non-shape function terms equal to 0
    check_name = 'N matrix zero-valued elements'
    _print_checking(check_name)
    check = np.all([np.all(N[:, :, 0, i - 1] == 0) for i in np.arange(nnodes, step=2)])
    _print_err_msg(check, err_msg=f'N matrix has too many nonzero values')

    # Verify components of N vs. individually calculated values
    check_name = 'N matrix components equivalence with corresponding nodal shape function'
    _print_checking(check_name)
    N_i = [sf(natural_grid) for sf in elem.get_shape_funcs()]
    for i, sf_vals in enumerate(N_i):
        check = np.all(N[:, :, 0, i*2] ==  sf_vals)
        _print_err_msg(check, err_msg=f'shape function {i} is not equal to the [0, {i}] component of matrix N')

    # Verify the first and second rows of are equal once shifted
    check_name = 'N matrix row equivalence'
    _print_checking(check_name)
    r1 = N[:, :, 0, :]
    r2 = np.roll(N[:, :, 1, :], -1)
    check = np.all(r1 == r2)
    _print_err_msg(check, err_msg=f'N matrix row 1 does not equal row 2 (shifted by -1)')

    # Success!
    _print_test_pass('shape function matrix (N)')

def test_2D_jacobian(elem: mfe.baseclasses.Element2D, ngrid: int = 50) -> None:
    test_name = '2D full Jacobian matrix (J) verification'
    _print_test_name(test_name)

    # Create a grid of coordinates in the natural coordinate system
    natural_grid = mfe.utils.make_natural_grid(ngrid)  

    # Compute the shape function derivative matrix dN and full Jacobian J
    dN = elem.compute_dN(natural_grid)
    J = elem.compute_J(dN)

    # Verify shape
    check_name = 'shape of full J matrix'
    _print_checking(check_name)
    nnodes = len(elem.nodes)
    check = J.shape[-2:] == (4, 4)
    _print_err_msg(check, err_msg=f'shape of full Jacobian matrix for 2D element is not (4, 4)')

    # Verify upper left quadrant of J equal to lower right
    check_name = 'upper left, lower right quadrant of full J matrix equivalence'
    _print_checking(check_name)
    check = np.all(J[:, :, 0:2, 0:2] == J[:, :, 2:4, 2:4])
    _print_err_msg(check, err_msg=f'full Jacobian upper left quadrant does not equal the lower right')

    # Verify upper right quadrant of J equal to lower left
    check_name = 'upper right, lower left quadrant of full J matrix equivalence'
    _print_checking(check_name)
    check = np.all(J[:, :, 0:2, 2:4] == J[:, :, 2:4, 0:2])
    _print_err_msg(check, err_msg=f'full J matrix upper right quadrant does not equal the lower left')

    # Verify upper right quadrant of J equal to zero matrix
    check_name = 'upper right quadrant of full J matrix equivalence with zero matrix'
    _print_checking(check_name)
    check = np.all(J[:, :, 0:2, 2:4] == np.zeros((2, 2)))
    _print_err_msg(check, err_msg=f'full J matrix upper right quadrant does not the zero matrix')

    # Verify full Jacobian when computed using natural nodal coordinates is the identity matrix
    check_name = 'full J matrix equal to the identity matrix when assembled from nodal coordinates in the natural coordinate system'
    _print_checking(check_name)
    q = mfe.utils.to_col_vec(elem.x_natural)
    q = mfe.utils.broadcast_ndarray_for_vectorziation(q, ngrid)
    J_col = np.matmul(dN, q)
    J_mat = np.array(
        [
            [J_col[:, :, 0, 0], J_col[:, :, 2, 0]], 
            [J_col[:, :, 1, 0], J_col[:, :, 3, 0]]
        ]
    )
    J_natural = elem._assemble_J(J_mat)
    check = np.all((J_natural - np.eye(4)) <= ERR_TOL)
    _print_err_msg(check, err_msg=f'full Jacobian computed in the natural coordiante system is not the identity matrix')

    # Success!
    _print_test_pass(test_name)

def test_2D_shape_function_derivatives(elem: mfe.baseclasses.Element2D, ngrid: int = 50) -> None:
    test_name = '2D shape function derivative matrix (dN) verification'
    _print_test_name(test_name)

    # Create a grid of coordinates in the natural coordinate system
    natural_grid = mfe.utils.make_natural_grid(ngrid)  

    # Compute the shape function derivative matrix dN and full Jacobian J
    dN = elem.compute_dN(natural_grid)

    # Verify shape
    check_name = 'shape of dN matrix'
    _print_checking(check_name)
    nnodes = len(elem.nodes)
    check = dN.shape[-2:] == (4, 2*nnodes)
    _print_err_msg(check, err_msg=f'shape of dN matrix for 2D element is not (4, {2*nnodes})')

    # Verify sum of shape function derivatives equal to 0
    check_name = 'sum of shape function derivatives'
    _print_checking(check_name)
    check = np.all(np.abs(np.sum(dN, axis=3)) <= ERR_TOL)
    _print_err_msg(check, err_msg=f'sum of shape function derivatives not equal to 0')

    # Verify non-shape function derivative terms equal to 0
    check_name = 'dN matrix zero-valued elements'
    _print_checking(check_name)
    check = np.all([np.all(dN[:, :, 0, i - 1] == 0) for i in np.arange(nnodes, step=2)])
    _print_err_msg(check, err_msg=f'shape function derivative matrix dN has too many nonzero values')

    # Verify components of dN vs. individually calculated values
    check_name = 'dN matrix components equivalence with corresponding nodal shape function derivatives in each coordinate direction'
    _print_checking(check_name)
    dN_i_1 = [dsf(natural_grid) for dsf in elem.get_shape_func_derivatives()[0]]
    dN_i_2 = [dsf(natural_grid) for dsf in elem.get_shape_func_derivatives()[1]]
    for i, (dsf_vals_1, dsf_vals_2) in enumerate(zip(dN_i_1, dN_i_2)):
        check = np.all(dN[:, :, 0, i*2] ==  dsf_vals_1)
        _print_err_msg(check, err_msg=f'shape function derivative dN{i}_1 is not equal to the [0, {i}] component of matrix dN')
        check = np.all(dN[:, :, 1, i*2] ==  dsf_vals_2)
        _print_err_msg(check, err_msg=f'shape function derivative dN{i}_2 is not equal to the [1, {i}] component of matrix dN')

    # Verify the first and third rows; second and fourth rows of are equal once shifted
    check_name = 'dN matrix row equivalence'
    _print_checking(check_name)
    r1 = dN[:, :, 0, :]
    r2 = dN[:, :, 1, :]
    r3 = np.roll(dN[:, :, 2, :], -1)
    r4 = np.roll(dN[:, :, 3, :], -1)
    check = np.all(r1 == r3)
    _print_err_msg(check, err_msg=f'shape function derivative matrix row 1 does not equal row 3 (shifted by -1)')
    check = np.all(r2 == r4)
    _print_err_msg(check, err_msg=f'shape function derivative matrix row 2 does not equal row 4 (shifted by -1)')

    # Success!
    _print_test_pass(test_name)

def test_2D_B_matrix(elem: mfe.baseclasses.Element2D, ngrid: int = 50) -> None:
    test_name = '2D B matrix verification'
    _print_test_name(test_name)

    # Create a grid of coordinates in the natural coordinate system
    natural_grid = mfe.utils.make_natural_grid(ngrid)  

    # Compute the shape function derivative matrix dN and full Jacobian J
    B = elem.compute_B(natural_grid)

    # Verify shape
    check_name = 'shape of B matrix'
    _print_checking(check_name)
    nnodes = len(elem.nodes)
    check = B.shape[-2:] == (3, 2*nnodes)
    _print_err_msg(check, err_msg=f'shape of dN matrix for 2D element is not (4, {2*nnodes})')

    # Verify non-shape function derivative terms equal to 0
    check_name = 'B matrix zero-valued elements'
    _print_checking(check_name)
    check = np.all([np.all(B[:, :, 0, i - 1] == 0) for i in np.arange(nnodes, step=2)])
    _print_err_msg(check, err_msg=f'B matrix has too many nonzero values in row 1')
    check = np.all([np.all(B[:, :, 1, i] == 0) for i in np.arange(nnodes, step=2)])
    _print_err_msg(check, err_msg=f'B matrix has too many nonzero values in row 2')

    # Verify components of B vs. 'hand-calculated' matrix math
    check_name = 'B matrix computation'
    _print_checking(check_name)
    dN = elem.compute_dN(natural_grid)
    J = elem.compute_J(dN)
    J_inv = np.linalg.inv(J)
    # Explicilty compute individual rows of B...
    r1 = np.array([J_inv[:, :, 0, 0]*dN[:, :, 0, i] + J_inv[:, :, 0, 1]*dN[:, :, 1, i] for i in np.arange(nnodes*2)])
    r2 = np.array([J_inv[:, :, 1, 0]*dN[:, :, 2, i] + J_inv[:, :, 1, 1]*dN[:, :, 3, i] for i in np.arange(nnodes*2)])
    r3 = np.array([
        J_inv[:, :, 1, 0]*dN[:, :, 0, i] + J_inv[:, :, 1, 1]*dN[:, :, 1, i]+ J_inv[:, :, 0, 0]*dN[:, :, 2, i]+ J_inv[:, :, 0, 1]*dN[:, :, 3, i]
        for i in np.arange(nnodes*2)
    ])
    for i, row in enumerate([r1, r2, r3]):
        row = mfe.utils.shift_ndarray_for_vectorization(row)
        check = np.all(B[:, :, i, :] == row)
        _print_err_msg(check, err_msg=f'B matrix row {i+1} computed through matrix math does not match explicit calculation')

    # Success!
    _print_test_pass(test_name)

def inspect_2D_element(
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
    u = elem.interpolate(nodal_vec, natural_grid)

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
        'disp': {
            'natural': mfe.plot.plot_element_displacement(u, natural_grid, method='scatter', coord_sys='natural'),
            'element': mfe.plot.plot_element_displacement(u, x_grid, method='scatter', coord_sys='element')
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
    ELEMENTS = {
        'linear': {
            'elem': mfe.elem_lib.Linear2D.from_element_coords(
                [
                    np.array([0, 0]), 
                    np.array([12, -1]), 
                    np.array([15, 8]), 
                    np.array([-1, 10])]
            ),
            'q': np.array([0, 0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])
        },
        'quad': {
            'elem': mfe.elem_lib.Quadratic2D.from_element_coords(
                [
                    np.array([0, 0]), 
                    np.array([6, 0.5]), 
                    np.array([12, -1]),
                    np.array([11, 5]),
                    np.array([15, 8]),
                    np.array([6, 11]),
                    np.array([-1, 10]),
                    np.array([1, 5]),
                ]
            ),
            'q': np.array([0, 0, 0.1, -0.1, 0.2, -0.3, 0.2, -0.3, 0.2, -0.3, 0.1, -0.1, 0.0, 0.1, 0.0, 0.0])
        },
    }
    MATERIALS = {
        'aluminum': mfe.baseclasses.Material(E=70000, nu=0.33)
    }
    elem = ELEMENTS['quad']['elem']
    nodal_disp = ELEMENTS['quad']['q']
    mat = mfe.baseclasses.Material(E=2500, nu=0.35)
    D = mat.D_isotropic_plane_stress()
    # inspect_2D_element(elem, D, nodal_disp)
    # test_2D_shape_functions(elem)
    # test_2D_jacobian(elem)
    # test_2D_shape_function_derivatives(elem)
    # test_2D_B_matrix(elem)
    natural_grid = mfe.utils.make_natural_grid(50)
    eta_1 = np.array([-0.577, 0.577]).reshape((2, 1))
    eta_2 = np.array([-0.577, 0.577]).reshape((2, 1))
    natural_grid = np.array(np.meshgrid(eta_1, eta_2)).reshape((2, 1, 2, 2))
    natural_grid = mfe.utils.shift_ndarray_for_vectorization(natural_grid)
    dN = elem.compute_dN(natural_grid)
    J = elem.compute_J(dN)
    detJ = np.linalg.det(J[:, :, 0:2, 0:2])
    print(detJ)
    print(natural_grid[0, 1])
    print(natural_grid[1, 1])
    # print(detJ[0, 0])
    # print(detJ[-1, -1])
    # print(detJ[5, 10])
    # print()

