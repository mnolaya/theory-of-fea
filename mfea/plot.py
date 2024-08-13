from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from mfea.elem_lib import Element2D    

def interpolate_element_grid(elem: Element2D, nvals: int, domain: str = 'element') -> tuple[np.ndarray]:
    '''
    Get coordinates within an element in terms of the natural or local element coordinate system.
    '''
    nodal_vec = {'element': elem.x_element, 'natural': elem.x_natural}
    return elem.interpolate(nodal_vec=nodal_vec[domain.lower()], nvals=nvals)

def _make_natural_grid(nvals: int) -> np.ndarray:
    '''
    Create a nvals x nvals grid for shape functions in the natural coordinate system.
    '''
    eta_1 = np.linspace(-1, 1, nvals)
    eta_2 = np.linspace(-1, 1, nvals)
    return np.meshgrid(eta_1, eta_2)

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

def plot_shape_functions(elem: Element2D, domain: str = 'natural', nvals: int = 100, nlevels: int = 50) -> None:
    '''
    Plot the the values of each shape function within an element in terms of the natural or local element coordinate system.
    '''
    # Create grid for the shape functions
    grid = _make_natural_grid(nvals)

    # Get element coordinates in natural or element coordinate system
    coords_x, coords_y = interpolate_element_grid(elem, nvals, domain)
    # if domain.lower() == 'element':
    #     labels = {'x': r'$\eta_{1}$', 'y': r'$\eta_{2}$'}
    #     coords_x, coords_y = grid
    # else:
    #     labels.update({'x': r'$x_{1}$', 'y': r'$x_{2}$'})
    #     coords_x, coords_y = self._compute_element_grid(domain, nvals)

    # Create figure and axes
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

    # Unravel axes into single list
    axes = [ax for _axes in axes for ax in _axes]

    # Loop through shape functions and plot contours
    plt_kw = {}
    shape_funcs = {i: f for i, f in enumerate(elem.get_shape_funcs(), start=1)}
    # shape_funcs = {1: self.shape_n1, 2: self.shape_n2, 3: self.shape_n3, 4: self.shape_n4}
    for ax, (i, func) in zip(axes, shape_funcs.items()):
        val = _compute_shape_func(func, grid)
        plt_kw.update({'levels': np.linspace(np.min(val), np.max(val), nlevels)})
        c = ax.contourf(coords_x, coords_y, val, **plt_kw)
        ax.set_title(rf'Shape function $N_{{{i}}}$')
        cbar = plt.colorbar(c, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        # ax.set_xlabel(labels['x'])
        # ax.set_ylabel(labels['y'])
    return fig, axes

# def plot_element_field(self, nodal_vec: np.ndarray, domain: str = 'natural', nvals: int = 100, nlevels: int = 50, field_label: str = r'$\phi$') -> None:
# # Get element coordinates in natural or element coordinate system
# _nodal_vec = self.x_natural
# labels = {'x': r'$\eta_{1}$', 'y': r'$\eta_{2}$'}
# if domain.lower() == 'element': 
#     _nodal_vec = self.x_element
#     labels.update({'x': r'$x_{1}$', 'y': r'$x_{2}$'})
# coords_x, coords_y = self.interpolate(nodal_vec=_nodal_vec, nvals=nvals)

# # Interpolate the field quantity
# vals_x, vals_y = self.interpolate(nodal_vec=nodal_vec, nvals=nvals)

# # Create figure and ax
# fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

# # Plot contours for interpolated field in each coordinate direction
# plt_kw = {'levels': np.linspace(np.min(vals_x), np.max(vals_x), nlevels)}
# c = axes[0].contourf(coords_x, coords_y, vals_x, **plt_kw)
# axes[0].set_title(rf'Field {field_label}$_{1}$')
# cbar = plt.colorbar(c, ticks=np.linspace(np.min(vals_x), np.max(vals_y), 5))
# axes[0].set_xlabel(labels['x'])
# axes[0].set_ylabel(labels['y'])
# plt_kw = {'levels': np.linspace(np.min(vals_y), np.max(vals_y), nlevels)}
# c = axes[1].contourf(coords_x, coords_y, vals_y, **plt_kw)
# axes[1].set_title(rf'Field {field_label}$_{2}$')
# cbar = plt.colorbar(c, ticks=np.linspace(np.min(vals_x), np.max(vals_y), 5))
# axes[1].set_xlabel(labels['x'])
# axes[1].set_ylabel(labels['y'])
# return fig, axes

