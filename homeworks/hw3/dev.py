import numpy as np
import matplotlib.pyplot as plt

import mfea.utils
from mfea.baseclasses import Material
from mfea.elem_lib import Linear2D, Quadratic2D
from mfea.plot import plot_shape_function, plot_element_field, plot_interpolated_element



def B_test() -> None:
    ngrid = 50
    elem = Linear2D.from_element_coords(
        [
            np.array([0, 0]), 
            np.array([12, -1]), 
            np.array([15, 8]), 
            np.array([-1, 10])
        ]
    )
    Q = np.array([0, 0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])
    mat = Material(E=2500, nu=0.35)
    D = mat.D_isotropic_plane_strain()

    eta = mfea.utils.make_natural_grid(ngrid)
    xgrid = elem.interpolate(eta, elem.x_element)
    ugrid = elem.interpolate(eta, Q)
    B = elem.B(eta)
    q = mfea.utils.broadcast_ndarray_for_vectorziation(mfea.utils.to_col_vec(Q), ngrid)
    eps = np.matmul(B, q)
    sigma = np.matmul(D, eps)

    fig, axes = plt.subplots(ncols=2)
    ax = plot_interpolated_element(
        xgrid, 
        ugrid[:, :, 0, 0], 
        ax=axes[0], 
        title=fr'$u_{1}$',
        # title=fr'$\varepsilon_{{{component}{component}}}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    ax = plot_interpolated_element(
        xgrid, 
        ugrid[:, :, 1, 0], 
        ax=axes[1], 
        title=fr'$u_{2}$',
        # title=fr'$\varepsilon_{{{component}{component}}}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    fig.tight_layout()

    dN = elem.dN(eta)
    J = elem.J(dN)
    fig, axes = plt.subplots(ncols=4)
    ax = plot_interpolated_element(
        xgrid, 
        J[:, :, 0, 0], 
        ax=axes[0], 
        title=fr'$J_{1}$',
        # title=fr'$\varepsilon_{{{component}{component}}}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    ax = plot_interpolated_element(
        xgrid, 
        J[:, :, 0, 1], 
        ax=axes[1], 
        title=fr'$J_{2}$',
        # title=fr'$\varepsilon_{{{component}{component}}}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    ax = plot_interpolated_element(
        xgrid, 
        J[:, :, 1, 0], 
        ax=axes[2], 
        title=fr'$J_{3}$',
        # title=fr'$\varepsilon_{{{component}{component}}}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    ax = plot_interpolated_element(
        xgrid, 
        J[:, :, 1, 1], 
        ax=axes[3], 
        title=fr'$J_{4}$',
        # title=fr'$\varepsilon_{{{component}{component}}}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    fig.tight_layout()

    print(np.mean(eps[:, :, 0, 0]))
    print(np.mean(eps[:, :, 1, 0]))
    print(np.mean(eps[:, :, 2, 0]))
    fig, axes = plt.subplots(ncols=3)
    ax = plot_interpolated_element(
        xgrid, 
        eps[:, :, 0, 0], 
        ax=axes[0], 
        # title=fr'$J_{1}$',
        title=r'$\varepsilon_{11}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    ax = plot_interpolated_element(
        xgrid, 
        eps[:, :, 1, 0], 
        ax=axes[1], 
        # title=fr'$J_{2}$',
        title=r'$\varepsilon_{22}$',
        levels=12,
        cmap='jet',
        continuous=True
    )
    ax = plot_interpolated_element(
        xgrid, 
        eps[:, :, 2, 0], 
        ax=axes[2], 
        # title=fr'$J_{3}$',
        title=r'$\varepsilon_{12}$',
        levels=12,
        cmap='jet',
        continuous=True
    )    
    fig.tight_layout()
    plt.show()

def grid_test() -> None:
    x = np.linspace(-1, 1, 1000)
    grid = np.meshgrid(x, x)
    # grid = np.array([0, 0])

    N1 = 0.25*(grid[0] - 1)*(grid[1] - 1)
    N2 = -0.25*(grid[0] + 1)*(grid[1] - 1)
    N3 = 0.25*(grid[0] + 1)*(grid[1] + 1)
    N4 = -0.25*(grid[0] - 1)*(grid[1] + 1)

    zs = np.zeros(N1.shape)
    N = np.array([[N1, zs, N2, zs, N3, zs, N4, zs], [zs, N1, zs, N2, zs, N3, zs, N4]])
    # fig, ax = plt.subplots()
    # ax.contourf(grid[0], grid[1], N[0, :, :])
    # plt.show()
    # print(test)

def N_test() -> None:
    elem = Linear2D.from_element_coords(
        [
            np.array([0, 0]), 
            np.array([12, -1]), 
            np.array([15, 8]), 
            np.array([-1, 10])
        ]
    )
    Q = np.array([0, 0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])

    fig, ax = plt.subplots()
    ax = plot_shape_function(elem, 3, coord_sys='natural')

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes = plot_element_field(elem, Q, axes=axes, coord_sys='natural')
    fig.tight_layout()
    plt.show()

    # phi = elem.interpolate(nodal_vec=elem.x_natural, ngrid=10)
    # fig, ax = plt.subplots()
    # ax.plot(phi[0, 0, :, :], phi[1, 0, :, :], marker='x', linestyle='none')
    # plt.show()
    # print(x_natural.shape)
    # grid = make_natural_grid(100)
    # N = elem.N(grid)
    # dN = elem.dN(grid)
    # fig, ax = plt.subplots()
    # ax.contourf(*grid, dN[0, 4, :, :])
    # plt.show()


if __name__ == '__main__':
    B_test()