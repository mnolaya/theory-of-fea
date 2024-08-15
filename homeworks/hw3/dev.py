import numpy as np
import matplotlib.pyplot as plt

from mfea.elem_lib import Linear2D, Quadratic2D
from mfea.plot import plot_shape_functions, plot_element_field

elem = Linear2D.from_element_coords(
    [
        np.array([0, 0]), 
        np.array([12, -1]), 
        np.array([15, 8]), 
        np.array([-1, 10])
    ]
)
Q = np.array([0, 0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])
# J = elem.J(Q, 0, 0)
B = elem.B(0, 0)
eps = elem.strain(0, 0)
print(eps)
# print(B)
# eps = np.matmul(B, Q)

