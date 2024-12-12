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

def apply_disp_bcs(x_disp: dict[int, float], y_disp: dict[int, float], K: np.ndarray, F: np.ndarray, penalty_scale: float = 1e6):
    # Create the penalty method stiffness scaled off the absolute maximum global stiffness
    C = np.max(np.abs(K))*penalty_scale

    # Loop through displacements in x and apply to K, F appropriately
    for node_num, disp in x_disp.items():
        idx = _get_node_matrix_index(node_num, 1, 2)
        K[idx, idx] = K[idx, idx] + C
        F[idx] = F[idx] + C*disp

    # Loop through displacements in y and apply to K, F appropriately
    for node_num, disp in y_disp.items():
        idx = _get_node_matrix_index(node_num, 2, 2)
        K[idx, idx] = K[idx, idx] + C
        F[idx] = F[idx] + C*disp
    
    return K, F

def build_assembly_coord_grid(G: np.ndarray, elems: list[baseclasses.Element2D], natural_grid: np.ndarray) -> np.ndarray:
    assembly_grid = []
    for i in range(G.shape[0]):
        elem_grid = elems[i].map_to_element(elems[i].x_global, natural_grid)
        assembly_grid.append(elem_grid)
    return np.vstack(assembly_grid)

def map_nodal_field_to_assembly(G: np.ndarray, elems: list[baseclasses.Element2D], Q: np.ndarray, natural_grid: np.ndarray, ndof: int = 2) -> tuple[np.ndarray]:
    assembly_field = []
    for i in range(G.shape[0]):
        node_field = []
        for j in range(G[i].shape[0]):
            for component in range(ndof):
                component += 1
                idx_row = _get_node_matrix_index(G[i][j], component, ndof)
                node_field.append(Q[idx_row, 0])
        node_field = np.array(node_field)
        assembly_field.append(elems[i].map_to_element(node_field, natural_grid))
    return np.vstack(assembly_field)

def map_stress_strain_to_assembly(G: np.ndarray, elems: list[baseclasses.Element2D], Q: np.ndarray, natural_grid: np.ndarray, ndof: int = 2, loc_sys: bool = True) -> tuple[np.ndarray]:
    stress_field = []
    strain_field = []
    for i in range(G.shape[0]):
        # Get nodal field for current element
        node_field = []
        for j in range(G[i].shape[0]):
            for component in range(ndof):
                component += 1
                idx_row = _get_node_matrix_index(G[i][j], component, ndof)
                node_field.append(Q[idx_row, 0])
        node_field = np.array(node_field)
        
        # Compute stress and strain
        elem = elems[i]
        dN = elem.compute_dN(natural_grid)
        J = elem.compute_J(dN)
        B = elem.compute_B(dN, J)
        strain = elem.compute_strain(B, node_field)
        stress = elem.compute_stress(elem.D, strain)

        if loc_sys:
            ER = baseclasses.EPS_TENS_TO_ENG_ROT
            strain = np.matmul(ER, np.matmul(elems[i].T, np.matmul(np.linalg.inv(ER), strain)))
            stress = np.matmul(elem.T, stress)

        strain_field.append(strain)
        stress_field.append(stress)
    return np.vstack(stress_field), np.vstack(strain_field)
