from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mfea.elem_lib import Element2D
from mfea.utils import make_natural_grid

CONTOUR_CMAP = 'jet'

def interpolate_element_grid(elem: Element2D, nvals: int, domain: str) -> tuple[np.ndarray]:
    '''
    Get coordinates within an element in terms of the natural or local element coordinate system.
    '''
    nodal_vec = {'element': elem.x_element, 'natural': elem.x_natural}
    return elem.interpolate(nodal_vec=nodal_vec[domain.lower()], nvals=nvals)

def _compute_shape_func(shape_func: Callable, eta_grid: tuple[np.ndarray]):
    '''
    Compute the value of a shape function for a collection of grid points in the natural coordinate system.
    '''
    val = []
    for i in range(eta_grid[0].shape[0]):
        row = []
        for j in range(eta_grid[1].shape[0]):
            e1, e2 = eta_grid[0][i, j], eta_grid[1][i][j]
            row.append(shape_func(e1, e2))
        val.append(np.array(row))
    return np.array(val)

def _get_ax_labels(domain: str) -> dict[str, str]:
    '''
    Get the x- and y-axes labels for plotting field quantities as a function of element position.
    '''
    labels = {
        'element': {'x': r'$x_{1}$', 'y': r'$x_{2}$'},
        'natural': {'x': r'$\eta_{1}$', 'y': r'$\eta_{2}$'},
    }
    return labels[domain.lower()]

def plot_shape_functions(elem: Element2D, domain: str = 'natural', nvals: int = 100, nlevels: int = 50) -> tuple[Figure, list[Axes]]:
    '''
    Plot the values of each shape function within an element in terms of the natural or local element coordinate system.
    '''
    # Create grid of natural coordinates across the element
    grid = make_natural_grid(nvals)

    # Get element coordinates in natural or element coordinate system
    coords_x, coords_y = interpolate_element_grid(elem, nvals, domain)
    ax_labels = _get_ax_labels(domain)

    # Create figure and axes
    ncols = int(len(elem.nodes)/2)
    fig, axes = plt.subplots(ncols=ncols, nrows=2, figsize=(ncols*5, 10))

    # Unravel axes into single list
    axes = [ax for _axes in axes for ax in _axes]

    # Loop through shape functions and plot contours
    plt_kw = {}
    shape_funcs = {i: f for i, f in enumerate(elem.get_shape_funcs(), start=1)}
    for ax, (i, func) in zip(axes, shape_funcs.items()):
        _plot_shape_function(ax, i, grid, coords_x, coords_y, func, ax_labels, nlevels)
        # val = _compute_shape_func(func, grid)
        # plt_kw.update({'levels': np.linspace(np.min(val), np.max(val), nlevels)})
        # c = ax.contourf(coords_x, coords_y, val, **plt_kw)
        # ax.set_title(rf'Shape function $N_{{{i}}}$')
        # ax.set_xlabel(ax_labels['x'])
        # ax.set_xlim(np.min(coords_x), np.max(coords_x))
        # ax.set_ylabel(ax_labels['y'])
        # ax.set_ylim(np.min(coords_y), np.max(coords_y))
        # cbar = plt.colorbar(c, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    return fig, axes

def plot_element_field(elem: Element2D, nodal_vec: np.ndarray, domain: str = 'natural', nvals: int = 100, nlevels: int = 50, field_label: str = r'$\phi$') -> None:
    '''
    Plot the interpolated field within an element given a vector of known nodal quantities.
    '''
    # Get element coordinates in natural or element coordinate system
    coords_x, coords_y = interpolate_element_grid(elem, nvals, domain)
    ax_labels = _get_ax_labels(domain)

    # Interpolate the field quantity
    vals_x, vals_y = elem.interpolate(nodal_vec=nodal_vec, nvals=nvals)
    mag = np.sqrt((vals_x**2) + (vals_y**2))

    # Create figure and axes
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    # Plot contours for interpolated field in each coordinate direction
    _plot_element_field(axes[0], 1, coords_x, coords_y, vals_x, ax_labels, field_label, nlevels)  # 1-component
    _plot_element_field(axes[1], 2, coords_x, coords_y, vals_y, ax_labels, field_label, nlevels)  # 2-component
    _plot_element_field(axes[2], 'magnitude', coords_x, coords_y, mag, ax_labels, field_label, nlevels)  # magnitude
    return fig, axes

def _plot_shape_function(ax: Axes, id: int | str, natural_grid: tuple[np.ndarray], coords_x: np.ndarray, coords_y: np.ndarray, func: Callable, ax_labels: dict[str, str], nlevels: int):
    val = _compute_shape_func(func, natural_grid)
    plt_kw = {'levels': np.linspace(np.min(val), np.max(val), nlevels), 'cmap': CONTOUR_CMAP}
    c = ax.contourf(coords_x, coords_y, val, **plt_kw)
    ax.set_title(rf'Shape function $N_{{{id}}}$')
    ax.set_xlabel(ax_labels['x'])
    ax.set_xlim(np.min(coords_x), np.max(coords_x))
    ax.set_ylabel(ax_labels['y'])
    ax.set_ylim(np.min(coords_y), np.max(coords_y))
    cbar = plt.colorbar(c, ticks=np.linspace(np.min(val), np.max(val), 5))

def _plot_element_field(ax: Axes, id: int | str, coords_x: np.ndarray, coords_y: np.ndarray, vals: np.ndarray, ax_labels: dict[str, str], field_label: str, nlevels: int):
    plt_kw = {'levels': np.linspace(np.min(vals), np.max(vals), nlevels), 'cmap': CONTOUR_CMAP}
    c = ax.contourf(coords_x, coords_y, vals, **plt_kw)
    ax.set_title(rf'Field {field_label}$_{{{id}}}$')
    cbar = plt.colorbar(c, ticks=np.linspace(np.min(vals), np.max(vals), 5))
    ax.set_xlabel(ax_labels['x'])
    ax.set_ylabel(ax_labels['y'])

