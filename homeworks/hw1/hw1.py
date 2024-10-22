from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import mfe.utils

def func_1(x: np.ndarray | float) -> np.ndarray | float:
    return 482.1*x**2 + x**3 - 42.356

def func_2(x: np.ndarray | float) -> np.ndarray | float:
    return np.array([x*3, x + 2])

def func_3(x: np.ndarray | float) -> np.ndarray | float:
    return np.array([x*15, x + 20, 0.5*x**3 - 10*x])

def func_4(x: np.ndarray) -> np.ndarray:
    # Compute A
    A = np.array(
        [
            [3*x[0, 0], 2*x[1, 0]],
            [x[1, 0] + x[1, 0], x[0, 0]**2 - 2]
        ]
    )
    # Reshape for vectorization
    x = mfe.utils.shift_ndarray_for_vectorization(x)
    A = mfe.utils.shift_ndarray_for_vectorization(A)

    # Solve
    return np.matmul(A, x)

def func_5(x: np.ndarray) -> np.ndarray:
    # Compute A
    A = np.array(
        [
            [3*x[0, 0], 2*x[1, 0]],
            [x[0, 0] + x[1, 0], x[0, 0]**2 - 2],
            [3/x[1, 0], x[0, 0]],
        ]
    )
    # Reshape for vectorization
    x = mfe.utils.shift_ndarray_for_vectorization(x)
    A = mfe.utils.shift_ndarray_for_vectorization(A)

    # Solve
    return np.matmul(A, x)

def plotter(func: Callable, xdata: np.ndarray) -> None:
    if len(xdata.shape) < 1: 
        print('error: xdata must be passed as a numpy array')
        exit()
    # Ensure xdata is a column vector [n x 1] or a grid of column vectors [n x 1 x grid_size_1 x grid_size_2]
    xdata = mfe.utils.to_col_vec(xdata)

    # Call the injected function
    ydata = func(xdata)

    # Plot depending on shape of data
    if len(ydata.shape) == 2:
        # Line plot
        fig, ax = plt.subplots()
        ax.plot(xdata, ydata, label='y')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.legend()
        fig.tight_layout()
    elif len(ydata.shape) == 3:
        # Line plot with more than one set of ydata
        fig, ax = plt.subplots()
        for i in range(ydata.shape[0]):
            ax.plot(xdata, ydata[i, ...], label=f'y{i+1}')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.legend()
            fig.tight_layout()
    elif len(ydata.shape) == 4:
        # Contour plots (more than one set of 2D ydata)
        for i in range(ydata.shape[2]):
            fig, ax = plt.subplots()
            ax.set_title(f'y{i+1}')
            ax.set_xlabel(r'$x_{1}$')
            ax.set_ylabel(r'$x_{2}$')
            c = ax.contourf(xdata[0, 0, ...], xdata[1, 0, ...], ydata[..., i, 0])
            plt.colorbar(c)
            fig.tight_layout()

if __name__ == '__main__':
    x = np.array(np.meshgrid(np.linspace(0, 10), np.linspace(0, 20)))
    plotter(func_4, x)
    plt.show()