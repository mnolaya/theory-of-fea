from typing import Callable
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import AxesImage
import matplotlib.colorbar

from mfe.elem_lib import Element2D
import mfe.utils

CONTOUR_CMAP = 'jet'

def _get_ax_labels(domain: str) -> dict[str, str]:
    '''
    Get the x- and y-axes labels for plotting field quantities as a function of element position.
    '''
    labels = {
        'element': {'x': r'$x_{1}$', 'y': r'$x_{2}$'},
        'natural': {'x': r'$\eta_{1}$', 'y': r'$\eta_{2}$'},
    }
    return labels[domain.lower()]

# def plot_shape_function(elem: Element2D, num: int, ax: Axes | None = None, coord_sys: str = 'natural', ngrid: int = 10, nlevels: int = 50) -> Axes:
#     # Create grid for the shape functions
#     eta = mfe.utils.make_natural_grid(ngrid)

#     # Interpolate for positions on the grid in either element or natural coordinates
#     if coord_sys.lower() == 'natural':
#         coords = elem.interpolate(eta=eta, nodal_vec=elem.x_natural)
#     else:
#         coords = elem.interpolate(eta=eta, nodal_vec=elem.x_element)
#     coords_x, coords_y = coords[0, 0, :, :], coords[1, 0, :, :]

#     # Compute N for the grid
#     N = elem.N(eta)

#     # Get the requested shape function values
#     N_i = mfe.utils.get_shape_func_by_num(N, num)

#     # Plot
#     if ax is None: ax = plt.gca()
#     plt_kw = {'levels': np.linspace(np.min(N_i), np.max(N_i), nlevels), 'cmap': CONTOUR_CMAP}
#     c = ax.contourf(coords_x, coords_y, N_i, **plt_kw)
#     ax_labels = _get_ax_labels(coord_sys)
#     ax.set_xlabel(ax_labels['x'])
#     ax.set_xlim(np.min(coords_x), np.max(coords_x))
#     ax.set_ylabel(ax_labels['y'])
#     ax.set_ylim(np.min(coords_y), np.max(coords_y))
#     cbar = plt.colorbar(c, ticks=np.linspace(np.min(N_i), np.max(N_i), 5))
#     return ax

# def plot_element_field(elem: Element2D, nodal_vec: np.ndarray, axes: np.ndarray[Axes], coord_sys: str = 'natural', ngrid: int = 10, nlevels: int = 50, field_label: str = r'$\phi$') -> np.ndarray[Axes]:
#     '''
#     Plot the interpolated field within an element given a vector of known nodal quantities.
#     '''
#     # Create grid for the shape functions
#     eta = mfe.utils.make_natural_grid(ngrid)

#     # Interpolate for positions on the grid in either element or natural coordinates
#     if coord_sys.lower() == 'natural':
#         coords = elem.interpolate(eta=eta, nodal_vec=elem.x_natural)
#     else:
#         coords = elem.interpolate(eta=eta, nodal_vec=elem.x_element)
#     coords_x, coords_y = coords[0, 0, :, :], coords[1, 0, :, :]

#     # Interpolate the field quantity
#     phi = elem.interpolate(eta, nodal_vec)
#     # mag = np.sqrt((vals_x**2) + (vals_y**2))

#     # Plot
#     ax_labels = _get_ax_labels(coord_sys)
#     _plot_element_field(axes[0], 1, coords_x, coords_y, phi[0, 0, :, :], ax_labels, field_label, nlevels)  # 1-component
#     _plot_element_field(axes[1], 2, coords_x, coords_y, phi[1, 0, :, :], ax_labels, field_label, nlevels)  # 2-component
#     # _plot_element_field(axes[2], 'magnitude', coords_x, coords_y, mag, ax_labels, field_label, nlevels)  # magnitude
#     # return fig, axes
#     return axes

# def _plot_element_field(ax: Axes, id: int | str, coords_x: np.ndarray, coords_y: np.ndarray, vals: np.ndarray, ax_labels: dict[str, str], field_label: str, nlevels: int):
#     plt_kw = {'levels': np.linspace(np.min(vals), np.max(vals), nlevels), 'cmap': CONTOUR_CMAP}
#     c = ax.contourf(coords_x, coords_y, vals, **plt_kw)
#     ax.set_title(rf'Field {field_label}$_{{{id}}}$')
#     cbar = plt.colorbar(c, ticks=np.linspace(np.min(vals), np.max(vals), 5))
#     ax.set_xlabel(ax_labels['x'])
#     ax.set_ylabel(ax_labels['y'])

def plot_interpolated_element(
        grid_coords: np.ndarray, 
        interpolated_field: np.ndarray,
        ax: Axes | None = None,
        title: str = '', 
        method: str = 'contour',
        levels: int = 10,
        coord_sys: str = 'element',
        cmap: str = 'jet',
        continuous: bool = False,
        clevels: int = 100,
    ) -> Axes:
    plt_kw = {'levels': levels, 'cmap': cmap}
    if ax is None: ax = plt.gca()
    x, y = grid_coords[:, :, 0, 0], grid_coords[:, :, 1, 0]
    if continuous: plt_kw.update({'levels': clevels})
    if method.lower() == 'contour': 
        c = ax.contourf(x, y, interpolated_field, **plt_kw)
    elif method.lower() == 'scatter':
        c = ax.scatter(x, y, c=interpolated_field, cmap=cmap)
    ticks = np.linspace(np.min(interpolated_field), np.max(interpolated_field), levels+1)
    cbar = plt.colorbar(c, ticks=ticks, format='{x:.2e}')
    cbar.ax.tick_params(labelsize=8)
    if title: ax.set_title(title)
    ax_labels = _get_ax_labels(coord_sys)
    ax.set_xlabel(ax_labels['x'])
    ax.set_ylabel(ax_labels['y'])
    return ax

def plot_element_stress_strain(
    stress: np.ndarray,
    strain: np.ndarray,
    grid_coords: np.ndarray,
    coord_sys: str = 'element',
    method: str = 'contour',
    **kwargs
) -> dict[str, tuple[Figure, np.ndarray[Axes]]]:
    kw = {'coord_sys': coord_sys, 'levels': 10, 'cmap': 'jet', 'continuous': True, 'method': method}
    kw.update(**kwargs)

    # Plot all results
    ncols = 3
    figs = {f: plt.subplots(ncols=ncols, figsize=(ncols*5, 5)) for f in ['stress', 'strain']}
    components = {0: '11', 1: '22', 2: '12'}
    for i in components.keys():
        # Plot strain component
        fig, axes = figs['strain']
        plt.figure(fig)
        plot_interpolated_element(
            grid_coords, strain[:, :, i, 0],
            ax=axes[i],
            title=fr'$\varepsilon_{{{components[i]}}}$',
            **kw
        )
        # Plot stress component
        fig, axes = figs['stress']
        plt.figure(fig)
        plot_interpolated_element(
            grid_coords, stress[:, :, i, 0],
            ax=axes[i],
            title=fr'$\sigma_{{{components[i]}}}$',
            **kw
        )
    figs['strain'][0].tight_layout()
    figs['stress'][0].tight_layout()
    return figs

def plot_element_displacement(
    u: np.ndarray,
    grid_coords: np.ndarray,
    coord_sys: str = 'element',
    method: str = 'contour',
    **kwargs
) -> tuple[Figure, np.ndarray[Axes]]:
    kw = {'coord_sys': coord_sys, 'levels': 10, 'cmap': 'jet', 'continuous': True, 'method': method}
    kw.update(**kwargs)

    # Plot all results
    ncols = 2
    fig, axes = plt.subplots(ncols=ncols, figsize=(ncols*5, 5))
    components = {0: '1', 1: '2'}
    for i in components.keys():
        # Plot stress component
        plot_interpolated_element(
            grid_coords, u[:, :, i, 0],
            ax=axes[i],
            title=fr'u$_{{{components[i]}}}$',
            **kw
        )
    fig.tight_layout()
    return fig, axes

def plot_element_shape_functions(
    N: np.ndarray,
    grid_coords: np.ndarray,
    method: str = 'contour',
    coord_sys: str = 'element',
    **kwargs
) -> tuple[Figure, np.ndarray[Axes]]:
    kw = {'coord_sys': coord_sys, 'levels': 10, 'cmap': 'jet', 'continuous': True, 'method': method}
    kw.update(**kwargs)

    # Plot all results
    nfuncs = int(np.ceil(N.shape[-1]/2))
    if nfuncs <= 4:
        ncols = nfuncs
        nrows = 1
    else:
        ncols = int(np.ceil(nfuncs/2))
        nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5, nrows*5))
    try:
        axes = list(itertools.chain.from_iterable(axes.tolist()))
    except TypeError:
        pass
    for i in range(int(N.shape[-1]/2)):
        ax = axes[i]
        N_i = N[:, :, 0, i*2]
        plot_interpolated_element(
            grid_coords, 
            N_i,
            ax=ax,
            title=fr'N$_{{{i+1}}}$',
            **kw
        )
    fig.tight_layout()
    return fig, axes

def plot_element_Jacobian(
    J: np.ndarray,
    grid_coords: np.ndarray,
    method: str = 'contour',
    coord_sys: str = 'element',
    **kwargs
) -> tuple[Figure, np.ndarray[Axes]]:
    kw = {'coord_sys': coord_sys, 'levels': 10, 'cmap': 'jet', 'continuous': True, 'method': method}
    kw.update(**kwargs)

    # Plot all results
    ncomp = J.shape[-1]
    if ncomp == 4:
        fig, axes = plt.subplots(ncols=ncomp, figsize=(ncomp*5, 5))
        comps = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    for i in range(ncomp):
        ax = axes[i]
        J_i = J[:, :, *comps[i]]
        plot_interpolated_element(
            grid_coords, 
            J_i,
            ax=ax,
            title=fr'J$_{{{comps[i][0]+1}{comps[i][1]+1}}}$',
            **kw
        )
    fig.tight_layout()
    return fig, axes
    
def plot_element_stiffness(k: np.ndarray) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    c = ax.imshow(k, cmap='coolwarm')
    cbar = fig.colorbar(c, format='{x:.2e}')
    cbar.ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(0, k.shape[0], 2))
    ax.set_ylabel('$i$-index')
    ax.set_yticks(np.arange(0, k.shape[1], 2))
    ax.set_xlabel('$j$-index')
    ax.set_title(r'Element stiffness matrix $\mathbf{k}$')
    return fig, ax
    
def plot_element_strain_energy_density(
    psi: np.ndarray,
    grid_coords: np.ndarray,
    method: str = 'scatter',
    coord_sys: str = 'element',
    **kwargs
) -> tuple[Figure, np.ndarray[Axes]]:
    kw = {'coord_sys': coord_sys, 'levels': 10, 'cmap': 'jet', 'continuous': True, 'method': method}
    kw.update(**kwargs)

    # Plot all results
    fig, ax = plt.subplots()
    plot_interpolated_element(
        grid_coords, 
        psi, 
        ax, 
        coord_sys='element', 
        method='scatter', 
        title=r'Element strain energy density $\psi$', 
        **kwargs,
    )
    fig.tight_layout()
    return fig, ax

def plot_stiffness_heatmap(fig: Figure, ax: Axes, k: np.ndarray, label_every: int | None = None, **kwargs) -> matplotlib.colorbar.Colorbar:
    # Set defaults
    plt_kwargs = {'cmap': 'coolwarm'}
    plt_kwargs.update(**kwargs)
    
    # Create heatmap
    c = ax.imshow(k, **plt_kwargs)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(c, cax=cax, format='{x:.2e}')

    # Label axes ticks if desired
    if label_every is not None:
        ax.set_xticks(np.arange(0, k.shape[0], label_every))
        ax.set_yticks(np.arange(0, k.shape[1], label_every))

    # Label axes and titles
    ax.set_ylabel('$i$-index')
    ax.set_xlabel('$j$-index')
    ax.set_title('Stiffness matrix')
    return cbar