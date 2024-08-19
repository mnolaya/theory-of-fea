from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mfea.elem_lib import Element2D
import mfea.utils

CONTOUR_CMAP = 'jet'

# def interpolate_element_grid(elem: Element2D, nvals: int, domain: str) -> tuple[np.ndarray]:
#     '''
#     Get coordinates within an element in terms of the natural or local element coordinate system.
#     '''
#     nodal_vec = {'element': elem.x_element, 'natural': elem.x_natural}
#     return elem.interpolate(nodal_vec=nodal_vec[domain.lower()], nvals=nvals)

# def _compute_shape_func(shape_func: Callable, eta_grid: tuple[np.ndarray]):
#     '''
#     Compute the value of a shape function for a collection of grid points in the natural coordinate system.
#     '''
#     val = []
#     for i in range(eta_grid[0].shape[0]):
#         row = []
#         for j in range(eta_grid[1].shape[0]):
#             e1, e2 = eta_grid[0][i, j], eta_grid[1][i][j]
#             row.append(shape_func(e1, e2))
#         val.append(np.array(row))
#     return np.array(val)

def _get_ax_labels(domain: str) -> dict[str, str]:
    '''
    Get the x- and y-axes labels for plotting field quantities as a function of element position.
    '''
    labels = {
        'element': {'x': r'$x_{1}$', 'y': r'$x_{2}$'},
        'natural': {'x': r'$\eta_{1}$', 'y': r'$\eta_{2}$'},
    }
    return labels[domain.lower()]

def plot_shape_function(elem: Element2D, num: int, ax: Axes | None = None, coord_sys: str = 'natural', ngrid: int = 10, nlevels: int = 50) -> Axes:
    # Create grid for the shape functions
    eta = mfea.utils.make_natural_grid(ngrid)

    # Interpolate for positions on the grid in either element or natural coordinates
    if coord_sys.lower() == 'natural':
        coords = elem.interpolate(eta=eta, nodal_vec=elem.x_natural)
    else:
        coords = elem.interpolate(eta=eta, nodal_vec=elem.x_element)
    coords_x, coords_y = coords[0, 0, :, :], coords[1, 0, :, :]

    # Compute N for the grid
    N = elem.N(eta)

    # Get the requested shape function values
    N_i = mfea.utils.get_shape_func_by_num(N, num)

    # Plot
    if ax is None: ax = plt.gca()
    plt_kw = {'levels': np.linspace(np.min(N_i), np.max(N_i), nlevels), 'cmap': CONTOUR_CMAP}
    c = ax.contourf(coords_x, coords_y, N_i, **plt_kw)
    ax_labels = _get_ax_labels(coord_sys)
    ax.set_xlabel(ax_labels['x'])
    ax.set_xlim(np.min(coords_x), np.max(coords_x))
    ax.set_ylabel(ax_labels['y'])
    ax.set_ylim(np.min(coords_y), np.max(coords_y))
    cbar = plt.colorbar(c, ticks=np.linspace(np.min(N_i), np.max(N_i), 5))
    return ax

def plot_element_field(elem: Element2D, nodal_vec: np.ndarray, axes: np.ndarray[Axes], coord_sys: str = 'natural', ngrid: int = 10, nlevels: int = 50, field_label: str = r'$\phi$') -> np.ndarray[Axes]:
    '''
    Plot the interpolated field within an element given a vector of known nodal quantities.
    '''
    # Create grid for the shape functions
    eta = mfea.utils.make_natural_grid(ngrid)

    # Interpolate for positions on the grid in either element or natural coordinates
    if coord_sys.lower() == 'natural':
        coords = elem.interpolate(eta=eta, nodal_vec=elem.x_natural)
    else:
        coords = elem.interpolate(eta=eta, nodal_vec=elem.x_element)
    coords_x, coords_y = coords[0, 0, :, :], coords[1, 0, :, :]

    # Interpolate the field quantity
    phi = elem.interpolate(eta, nodal_vec)
    # mag = np.sqrt((vals_x**2) + (vals_y**2))

    # Plot
    ax_labels = _get_ax_labels(coord_sys)
    _plot_element_field(axes[0], 1, coords_x, coords_y, phi[0, 0, :, :], ax_labels, field_label, nlevels)  # 1-component
    _plot_element_field(axes[1], 2, coords_x, coords_y, phi[1, 0, :, :], ax_labels, field_label, nlevels)  # 2-component
    # _plot_element_field(axes[2], 'magnitude', coords_x, coords_y, mag, ax_labels, field_label, nlevels)  # magnitude
    # return fig, axes
    return axes

def _plot_element_field(ax: Axes, id: int | str, coords_x: np.ndarray, coords_y: np.ndarray, vals: np.ndarray, ax_labels: dict[str, str], field_label: str, nlevels: int):
    plt_kw = {'levels': np.linspace(np.min(vals), np.max(vals), nlevels), 'cmap': CONTOUR_CMAP}
    c = ax.contourf(coords_x, coords_y, vals, **plt_kw)
    ax.set_title(rf'Field {field_label}$_{{{id}}}$')
    cbar = plt.colorbar(c, ticks=np.linspace(np.min(vals), np.max(vals), 5))
    ax.set_xlabel(ax_labels['x'])
    ax.set_ylabel(ax_labels['y'])

def plot_interpolated_element(
        grid_coords: np.ndarray, 
        interpolated_field: np.ndarray,
        ax: Axes | None = None,
        title: str = '', 
        nticks: int = 5,
        levels: int = 50,
        coord_sys: str = 'element',
    ) -> Axes:
    if ax is None: ax = plt.gca()
    x, y = grid_coords[:, :, 0, 0], grid_coords[:, :, 1, 0]
    c = ax.contourf(x, y, interpolated_field, levels=levels)
    plt.colorbar(c, ticks=np.linspace(np.min(interpolated_field), np.max(interpolated_field), nticks))
    if title: ax.set_title(title)
    ax_labels = _get_ax_labels(coord_sys)
    ax.set_xlabel(ax_labels['x'])
    ax.set_ylabel(ax_labels['y'])
    return ax