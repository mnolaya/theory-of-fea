import io
import pathlib

import polars as pl
import numpy as np
import numpy.typing as npt

def make_natural_grid(ngrid: int = 50) -> np.ndarray:
    '''
    Create a ngrid x ngrid grid for shape functions in the natural coordinate system.
    '''
    dx = np.linspace(-1, 1, ngrid)
    grid = np.array(np.meshgrid(dx, dx)).reshape((2, 1, ngrid, ngrid))
    return shift_ndarray_for_vectorization(grid)

def get_shape_func_by_num(N: np.ndarray, i: int) -> np.ndarray:
    '''
    Get the shape function values by shape function number.
    '''
    sfunc_idx = {i + 1: -2 + 2*(i + 1) for i in range(int(N.shape[1]/2))}
    return N[0, sfunc_idx[i], :, :]

# def coords_as_grid(coord_vec: np.ndarray) -> np.ndarray:
#     coords_grid = np.zeros((coord_vec.shape[0]))
#     return 
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
    if not isinstance(arr, np.ndarray): arr = np.array(arr)
    if len(arr.shape) == 1:
        # Convert 1D to column vector [n x 1]
        return arr.reshape((arr.shape[0], 1))
    elif len(arr.shape) == 2 and arr.shape[1] == 1:
        # Already in correct form [n x 1]
        return arr
    elif len(arr.shape) == 3:
        # Reshape to grid of column vectors [n x 1 x grid_size_1 x grid_size_2]
        return arr.reshape((arr.shape[0], 1, *arr.shape[-2:]))
        
def shift_ndarray_for_vectorization(arr: np.ndarray) -> np.ndarray:
    '''
    Shift a numpy array of shape m x n x ngrid x ngrid to 
    ngrid x ngrid x m x n to facilitate vectorized matrix multiplication
    '''
    arr = np.swapaxes(arr, 0, -2)
    arr = np.swapaxes(arr, 1, -1)
    return arr

def broadcast_ndarray_for_vectorziation(arr: np.ndarray, grid_shape: tuple[int]) -> np.ndarray:
    '''
    Use numpy broadcasting to stack/duplicate a numpy array of shape m x n onto a
    a grid of ngrid x ngrid points to facilitate vectorized matrix multiplication
    '''
    return np.zeros(grid_shape)[:, :, np.newaxis, np.newaxis] + arr

def components_from_grid(grid: np.ndarray) -> tuple[np.ndarray]:
    return grid[:, :, 0, 0], grid[:, :, 1, 0]

def read_connectivity_from_csv(fp: pathlib.Path, revise: bool = False) -> np.ndarray:
    try:
        # If all elements are the same in terms of number of nodes, this should pass
        df = pl.read_csv(fp, has_header=False)
    except:
        # If not, append dummy commas to make the csv readable by polars
        with open(fp, 'r') as f:
            lines = f.readlines()
        # Count the max number of columns required in the csv format
        max_cols = 2
        ncols_per_line = []
        for line in lines:
            ncols = len(line.split(','))
            ncols_per_line.append(ncols)
            if ncols > max_cols: max_cols = ncols
        # Revise file lines with dummy commas
        rev_lines = []
        for (line, ncols) in zip(lines, ncols_per_line):
            diff = max_cols - ncols
            rev_line = line.strip()
            if diff != 0:
                rev_line += ','*diff # Append commas with empty values to end of line
            rev_lines.append(rev_line)
        # Read revised lines into csv, where dummy entries are filled with nan
        df = pl.read_csv(io.StringIO('\n'.join(rev_lines)), has_header=False)
    
    # Update file with dummy commas if requested
    if revise:
        with open(fp, 'w+') as f:
            f.writelines(rev_lines)

    # Return as numpy array
    return df.to_numpy()

def read_mesh_from_csv(connectivity: pathlib.Path, node_coords: pathlib.Path) -> tuple[np.ndarray]:
    # Read nodal coordinates
    node_coords_df = pl.read_csv(node_coords)

    # Read connectivity matrix
    G = read_connectivity_from_csv(connectivity)

    return G, node_coords_df.to_numpy()
    
def make_transform_matrix_2D(theta: float) -> np.ndarray:
    '''
    Make a 2D rotation matrix:
    [cos(theta), sin(theta)]
    [-sin(theta), cos(theta)]
    '''
    rad = np.deg2rad(theta)
    m = np.cos(rad)
    n = np.sin(rad)
    return np.array(
        [
            [m**2, n**2, 2*m*n],
            [n**2, m**2, -2*m*n],
            [-m*n, m*n, (m**2) - (n**2)]
        ],
    )

def transform_voigt_2D(theta: float, tens_voigt: np.ndarray) -> np.ndarray:
    '''
    Transform 2D Voigt notation matrix (3x1) according to some angle theta.
    '''
    T = make_transform_matrix_2D(theta)
    return np.matmul(T, tens_voigt)
    
def D_transversely_isotropic_plane_stress(E11, E22, nu12, G12) -> np.ndarray:
    '''
    Compute D for a transversely isotropic material.
    '''
    S = np.eye(3)
    S[0, 0] = 1/E11
    S[0, 1] = -nu12/E11
    S[1, 0] = S[0, 1]
    S[1, 1] = 1/E22
    S[2, 2] = 1/G12
    return np.linalg.inv(S)