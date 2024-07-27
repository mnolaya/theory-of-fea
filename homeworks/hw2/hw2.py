import numpy as np
import matplotlib.pyplot as plt
from attrs import define

@define
class Node:
    x: float
    y: float
    global_num: int
    elem_num: int

@define
class Linear2D:
    # nodes: list[Node]
    
    def shape_n1(self, eta_1: float, eta_2: float) -> float:
        return 0.25*(eta_1 - 1)*(eta_2 - 1)

    def shape_n2(self, eta_1: float, eta_2: float) -> float:
        return -0.25*(eta_1 + 1)*(eta_2 - 1)

    def shape_n3(self, eta_1: float, eta_2: float) -> float:
        return 0.25*(eta_1 + 1)*(eta_2 + 1)

    def shape_n4(self, eta_1: float, eta_2: float) -> float:
        return -0.25*(eta_1 - 1)*(eta_2 + 1)
    
    def plot_shape_functions(self, nvals: int = 1000) -> None:
        # Create grid for the shape functions
        eta_1 = np.linspace(-1, 1, nvals)
        eta_2 = np.linspace(-1, 1, nvals)
        eta_1, eta_2 = np.meshgrid(eta_1, eta_2)

        # Create figure and axes
        # fig, axes = plt.subplots(ncols=2, nrows=2)
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

        # Unravel axes into single list
        axes = [ax for _axes in axes for ax in _axes]

        # Loop through shape functions and plot contours
        shape_funcs = {1: self.shape_n1, 2: self.shape_n2, 3: self.shape_n3, 4: self.shape_n4}
        for ax, (i, func) in zip(axes, shape_funcs.items()):
            ax.hlines(0, -1, 1, color='tab:grey', lw=0.5, linestyle='--')
            ax.vlines(0, -1, 1, color='tab:grey', lw=0.5, linestyle='--')
            c = ax.contourf(eta_1, eta_2, func(eta_1, eta_2))
            ax.set_title(rf'Shape function $N_{{{i}}}$')
            ax.set_xlabel(r'$\eta_{1}$ position')
            ax.set_xlim(-1, 1)
            ax.set_ylabel(r'$\eta_{2}$ position')
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
        return fig, axes

    
element = Linear2D()
fig, axes = element.plot_shape_functions()

# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
# axes = [ax for _axes in axes for ax in _axes]
# for ax in axes:
#     ax.hlines(0, -1, 1, color='tab:grey', lw=0.5, linestyle='--')
#     ax.vlines(0, -1, 1, color='tab:grey', lw=0.5, linestyle='--')

# c = axes[0].contourf(eta_1, eta_2, element.shape_n1(eta_1, eta_2))
# axes[0].set_title(r'$N_{1}$')
# axes[0].set_xlabel(r'$\eta_{1}$ position')
# axes[0].set_ylabel(r'$\eta_{2}$ position')
# c = axes[1].contourf(eta_1, eta_2, element.shape_n2(eta_1, eta_2))
# axes[1].set_title(r'$N_{2}$')
# c = axes[2].contourf(eta_1, eta_2, element.shape_n3(eta_1, eta_2))
# axes[2].set_title(r'$N_{3}$')
# c = axes[3].contourf(eta_1, eta_2, element.shape_n4(eta_1, eta_2))
# axes[3].set_title(r'$N_{4}$')
# # fig.colorbar(c, ax=axes[2], fraction=0.02, pad=0.1)
fig.tight_layout()
# ax.set_aspect('equal')
# plt.show()

nodes = np.array([[0, 0], [12, -1], [15, 8], [-1, 10]])
x = np.array([
    element.shape_n1(nodes[0, 0], nodes[0, 1]),
    element.shape_n2(nodes[1, 0], nodes[1, 1]),
    element.shape_n3(nodes[2, 0], nodes[2, 1]),
    element.shape_n4(nodes[3, 0], nodes[3, 1]),
])
print(x)
