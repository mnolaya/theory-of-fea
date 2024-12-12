import pathlib

import polars as pl
import numpy as np

# Read nodes
# Read elements
# Read bcs

ABQ_MESH_BASE_DIR = pathlib.Path(r"C:\Users\Michael\Documents\repos\theory_fea\project\mesh")

def get_mesh(inp_lines: list[str], loc: str = 'node') -> list[str]:
    mesh = []
    gather = False
    for line in inp_lines:
        if gather and '*' in line:
            gather = False
            continue
        elif gather:
            mesh_line = [float(s.strip()) for s in line.split(',')[1:]]
            mesh.append(mesh_line)
        elif f'*{loc.lower().capitalize()}' in line:
            gather = True
    if loc.lower() == 'element':
        dtype = int
    else:
        dtype = float
    return np.array(mesh, dtype)

def get_bc_nodes(inp_lines: list[str], loc: str = 'bottom') -> list[str]:
    bc_nodes = []
    gather = False
    generate = False
    for line in inp_lines:
        if gather and '*' in line:
            gather = False
            continue
        elif gather:
            if generate:
                node_line = [int(s.strip()) for s in line.split(',')]
                node_line = np.arange(node_line[0], node_line[1]+1, node_line[2], dtype=int)
                break
            else:
                node_line = [int(s.strip()) for s in line.split(',')]
            bc_nodes.append(node_line)
        elif f'*Nset, nset={loc.lower()}' in line:
            gather = True
            if 'generate' in line: generate = True
    return np.array(bc_nodes)

def main() -> None:
    for fp in ABQ_MESH_BASE_DIR.rglob('*.inp'):
        print(f'converting abq mesh to mfe mesh -> {fp}')
        job_dir = fp.parent
        job_name = fp.stem
        with open(fp, 'r') as f:
            lines = f.readlines()

        # Convert nodes to mfe format
        nodes = get_mesh(lines, 'node')
        out = job_dir.joinpath(f'{job_name}_nodes_mfe.csv')
        pl.DataFrame(nodes, schema=['x', 'y']).write_csv(out)

        # Convert connectivity to mfe format
        elems = get_mesh(lines, 'element')
        out = job_dir.joinpath(f'{job_name}_connectivity_mfe.csv')
        pl.DataFrame(elems).write_csv(out, include_header=False)

        # Convert bc nodes to mfe format
        bcs_bot = get_bc_nodes(lines, 'bottom')
        bcs_top = get_bc_nodes(lines, 'top')
        out = job_dir.joinpath(f'{job_name}_bcnodes_mfe.csv')
        pl.DataFrame(np.vstack([bcs_bot, bcs_top]), schema=['bottom', 'top'], orient='col').write_csv(out)

if __name__ == "__main__":
    main()