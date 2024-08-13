# import typing
# import itertools

# import numpy as np
# import matplotlib.pyplot as plt
# from attrs import define

# # Compute the value of a shape function for a series of grid points in the natural coordinate system.
# def _compute_shape_func(shape_func: typing.Callable, eta_grid: tuple[np.ndarray]):
#     val = []
#     for i in range(eta_grid[0].shape[0]):
#         row = []
#         for j in range(eta_grid[1].shape[0]):
#             e1, e2 = eta_grid[0][i, j], eta_grid[1][i][j]
#             row.append(shape_func(e1, e2))
#         val.append(np.array(row))
#     return np.array(val)

# @define
# class Node:
#     local_num: int
#     local_coords: np.ndarray
#     natural_coords: np.ndarray

# @define
# class Linear2D:
#     nodes: list[Node]

#     @classmethod
#     def from_element_coords(cls, coords: list[np.ndarray]):
#         nodes = [
#             Node(local_num=1, local_coords=coords[0], natural_coords=np.array([-1, -1])),
#             Node(local_num=2, local_coords=coords[1], natural_coords=np.array([1, -1])),
#             Node(local_num=3, local_coords=coords[2], natural_coords=np.array([1, 1])),
#             Node(local_num=4, local_coords=coords[3], natural_coords=np.array([-1, 1])),
#         ]
#         return cls(nodes)

#     def N(self, eta_1: float, eta_2: float) -> np.ndarray:
#         return [
#             [self.shape_n1(eta_1, eta_2), 0, self.shape_n2(eta_1, eta_2), 0, self.shape_n3(eta_1, eta_2), 0, self.shape_n4(eta_1, eta_2), 0],
#             [0, self.shape_n1(eta_1, eta_2), 0, self.shape_n2(eta_1, eta_2), 0, self.shape_n3(eta_1, eta_2), 0, self.shape_n4(eta_1, eta_2)],
#         ]

#     @property
#     def x_natural(self) -> np.ndarray:
#         return list(itertools.chain.from_iterable([node.natural_coords for node in self.nodes]))

#     @property
#     def x_element(self) -> np.ndarray:
#         return list(itertools.chain.from_iterable([node.local_coords for node in self.nodes]))
    
#     def get_shape_funcs(self) -> list:
#         return [self.shape_n1, self.shape_n2, self.shape_n3, self.shape_n4]
    
#     @staticmethod
#     def shape_n1(eta_1: float, eta_2: float) -> float:
#         return 0.25*(eta_1 - 1)*(eta_2 - 1)

#     @staticmethod
#     def shape_n2(eta_1: float, eta_2: float) -> float:
#         return -0.25*(eta_1 + 1)*(eta_2 - 1)

#     @staticmethod
#     def shape_n3(eta_1: float, eta_2: float) -> float:
#         return 0.25*(eta_1 + 1)*(eta_2 + 1)

#     @staticmethod
#     def shape_n4(eta_1: float, eta_2: float) -> float:
#         return -0.25*(eta_1 - 1)*(eta_2 + 1)
    
#     def interpolate(self, nodal_vec: np.ndarray, nvals: int = 100) -> tuple[np.ndarray]:
#         # Create grid for the shape functions
#         eta_1 = np.linspace(-1, 1, nvals)
#         eta_2 = np.linspace(-1, 1, nvals)
#         eta_1, eta_2 = np.meshgrid(eta_1, eta_2)

#         # Interpolate for each point on the grid
#         interpolated = []
#         for i in range(nvals):
#             row = []
#             for j in range(nvals):
#                 e1, e2 = eta_1[i, j], eta_2[i, j]
#                 row.append(np.matmul(self.N(e1, e2), nodal_vec))
#             interpolated.append(np.array(row))
#         interpolated = np.array(interpolated)
#         return interpolated[:, :, 0], interpolated[:, :, 1]  

#     def plot_shape_functions(self, domain: str = 'natural', nvals: int = 100, nlevels: int = 50) -> None:
#         # Get element coordinates in natural or element coordinate system
#         nodal_vec = self.x_natural
#         labels = {'x': r'$\eta_{1}$', 'y': r'$\eta_{2}$'}
#         if domain.lower() == 'element': 
#             nodal_vec = self.x_element
#             labels.update({'x': r'$x_{1}$', 'y': r'$x_{2}$'})
#         coords_x, coords_y = self.interpolate(nodal_vec=nodal_vec, nvals=nvals)

#         # Create grid for the shape functions
#         eta_1 = np.linspace(-1, 1, nvals)
#         eta_2 = np.linspace(-1, 1, nvals)
#         grid = np.meshgrid(eta_1, eta_2)

#         # Create figure and axes
#         fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

#         # Unravel axes into single list
#         axes = [ax for _axes in axes for ax in _axes]

#         # Loop through shape functions and plot contours
#         plt_kw = {}
#         shape_funcs = {1: self.shape_n1, 2: self.shape_n2, 3: self.shape_n3, 4: self.shape_n4}
#         for ax, (i, func) in zip(axes, shape_funcs.items()):
#             val = _compute_shape_func(func, grid)
#             plt_kw.update({'levels': np.linspace(np.min(val), np.max(val), nlevels)})
#             c = ax.contourf(coords_x, coords_y, val, **plt_kw)
#             ax.set_title(rf'Shape function $N_{{{i}}}$')
#             cbar = plt.colorbar(c, ticks=[0, 0.25, 0.5, 0.75, 1.0])
#             ax.set_xlabel(labels['x'])
#             ax.set_ylabel(labels['y'])
#         return fig, axes
    
#     def plot_element_field(self, nodal_vec: np.ndarray, domain: str = 'natural', nvals: int = 100, nlevels: int = 50, field_label: str = r'$\phi$') -> None:
        # Get element coordinates in natural or element coordinate system
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
import numpy as np
import matplotlib.pyplot as plt

from mfea.elem_lib import Linear2D
from mfea.plot import plot_shape_functions


def main(q: np.ndarray, nvals: int) -> None:
    element = Linear2D.from_element_coords(
        [np.array([0, 0]), np.array([12, -1]), np.array([15, 8]), np.array([-1, 10])]
    )

    fig, axes = plot_shape_functions(element, domain='natural', nvals=nvals, nlevels=50)
    fig.tight_layout()

    # fig, axes = element.plot_shape_functions(domain='element', nvals=nvals, nlevels=50)
    # fig.tight_layout()

    # fig, axes = element.plot_element_field(nodal_vec=q, domain='element', nvals=nvals, nlevels=50, field_label='u')
    # fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    q = np.array([0, 0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])
    nvals = 100
    main(q, nvals)