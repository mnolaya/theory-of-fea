import numpy as np
import numpy.typing as npt

def make_natural_grid(ngrid: int) -> np.ndarray:
    '''
    Create a ngrid x ngrid grid for shape functions in the natural coordinate system.
    '''
    dx = np.linspace(-1, 1, ngrid)
    return np.array(np.meshgrid(dx, dx))

def get_shape_func_by_num(N: np.ndarray, i: int) -> np.ndarray:
    '''
    Get the shape function values by shape function number.
    '''
    sfunc_idx = {i + 1: -2 + 2*(i + 1) for i in range(int(N.shape[1]/2))}
    return N[0, sfunc_idx[i], :, :]

# def shift_ndarray_val_axis(arr: np.ndarray) -> np.ndarray:
#     '''
#     Shift the axis of a i x j x ngrid x ngrid numpy array where values are found from the first two
#     to the last two indices to facilitate vectorized matrix multiplication.
#     '''
#     return np.moveaxis(arr, [0, 1], [-2, -1])

# def vectorize_nodal_vec(col_vec: np.ndarray, shape: tuple[int]) -> np.ndarray:
#     '''
#     Use numpy broadcasting to assemble a matrix of identical column vectors, one for each
#     point on a ngrid x ngrid sized grid to facilitate vectorized matrix multiplication.
#     '''
#     # col_vec = col_vec.reshape((col_vec.shape[0], 1))  # Ensure input vector is a column vector
#     return col_vec[:, np.newaxis, np.newaxis] + np.zeros(shape)

def to_col_vec(arr: npt.ArrayLike) -> np.ndarray:
    arr = np.array(arr)
    return arr.reshape((arr.shape[0], 1))

def shift_ndarray_for_vectorization(arr: np.ndarray) -> np.ndarray:
    '''
    Shift a numpy array of shape m x n x ngrid x ngrid to 
    ngrid x ngrid x m x n to facilitate vectorized matrix multiplication
    '''
    return np.moveaxis(arr, [0, 1], [-2, -1])

def broadcast_ndarray_for_vectorziation(arr: np.ndarray, ngrid: int) -> np.ndarray:
    '''
    Use numpy broadcasting to stack/duplicate a numpy array of shape m x n onto a
    a grid of ngrid x ngrid points to facilitate vectorized matrix multiplication
    '''
    return np.zeros((ngrid, ngrid))[:, :, np.newaxis, np.newaxis] + arr