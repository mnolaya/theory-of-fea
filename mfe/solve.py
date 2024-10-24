import numpy as np

from mfe import elem_lib
from mfe import baseclasses
from mfe import load

ELEMENT_BY_NODES = {
    4: elem_lib.Linear2D,
    8: elem_lib.Quadratic2D
}

def _get_node_matrix_index(node_num: int, component: int, ndof: int) -> int:
    return ndof*(node_num - 1) + component - 1

def assemble_mesh(G: np.ndarray, node_coords: np.ndarray) -> list[baseclasses.Element2D]:
    '''
    Build a mesh from connectivity matrix and nodal coordinates.
    '''
    elems = []
    for global_nodes in G:
        idx_slice = [i-1 for i in global_nodes.tolist()]
        elem = ELEMENT_BY_NODES[len(idx_slice)]
        elem_coords = node_coords[idx_slice, ...]
        elems.append(elem.from_element_coords(elem_coords))
    return elems

def assemble_global_solution(G: np.ndarray, elems: list[baseclasses.Element2D], loads: list[load.SurfaceTraction], ndof: int = 2) -> tuple[np.ndarray]:
    # Get the total number of nodes in the model
    nnodes = int(np.nanmax(G))

    # Initialize global stiffness [K] and global force vector [F]
    K = np.zeros((ndof*nnodes, ndof*nnodes))
    F = np.zeros((ndof*nnodes, 1))

    ## Assemble

    for i in range(G.shape[0]):
        # Get element connectivity row
        elem_connect = G[i]

        # Compute the local element stiffness matrix and force vector
        k_e = elems[i].compute_k()
        f_e = np.zeros((ndof*elems[i].nnodes, 1))
        if loads[i]: f_e = loads[i].compute_force_vector(elems[i])

        for j in range(elem_connect.shape[0]):
            if np.isnan(elem_connect[j]): continue  # Skip any nan rows (filled by numpy for dissimilar elements in terms of number of nodes)

            # Get current local element and corresponding global node numbers for row j of the global solution
            local_node_row = j + 1  # Python indexing starts at 0; add 1
            global_node_row = elem_connect[j]

            for component in range(ndof):
                component += 1 # Python indexing starts at 0; add 1

                # Get local element and corresponding global row index for assembly
                local_row_idx = _get_node_matrix_index(local_node_row, component, ndof)
                global_row_idx = _get_node_matrix_index(global_node_row, component, ndof)

                # Update global force vector
                F[global_row_idx, 0] = F[global_row_idx, 0] + f_e[local_row_idx, 0]

                for k in range(elem_connect.shape[0]):
                    # Get current local element and corresponding global node numbers for col k of the global solution
                    local_node_col = k + 1
                    global_node_col = elem_connect[k]

                    for component in range(ndof):
                        component += 1 # Python indexing starts at 0; add 1

                        # Get local element and corresponding global col index for assembly
                        local_col_idx = _get_node_matrix_index(local_node_col, component, ndof)
                        global_col_idx = _get_node_matrix_index(global_node_col, component, ndof)

                        # Update global stiffness matrix
                        K[global_row_idx, global_col_idx] = K[global_row_idx, global_col_idx] + k_e[local_row_idx, local_col_idx]
    return K, F

