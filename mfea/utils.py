import numpy as np

def make_natural_grid(nvals: int) -> np.ndarray:
    '''
    Create a nvals x nvals grid for shape functions in the natural coordinate system.
    '''
    eta_1 = np.linspace(-1, 1, nvals)
    eta_2 = np.linspace(-1, 1, nvals)
    return np.meshgrid(eta_1, eta_2)