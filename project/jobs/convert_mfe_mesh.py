import pathlib

import polars as pl
import numpy as np

# Read nodes
# Read elements
# Read bcs

ABQ_MESH_BASE_DIR = pathlib.Path(r"C:\Users\Michael\Documents\repos\theory_fea\project\jobs\linear")

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

def get_nset_nodes(inp_lines: list[str], setname: str = 'bottom') -> list[str]:
    nset_nodes = []
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
                nset_nodes.append(node_line)
                break
            else:
                node_line = np.array([int(i) for i in [s.strip() for s in line.split(',')] if i])
            nset_nodes.append(node_line)
        elif f'*Nset, nset={setname.lower()}' in line:
            gather = True
            if 'generate' in line: generate = True
    return np.array(np.concatenate(nset_nodes))

def get_elset_elems(inp_lines: list[str], setname: str = 'bottom') -> list[str]:
    elset_nodes = []
    gather = False
    generate = False
    for line in inp_lines:
        if gather and '*' in line:
            gather = False
            continue
        elif gather:
            if generate:
                elem_line = [int(s.strip()) for s in line.split(',')]
                elem_line = np.arange(elem_line[0], elem_line[1]+1, elem_line[2], dtype=int)
                elset_nodes.append(elem_line)
                break
            else:
                elem_line = np.array([int(i) for i in [s.strip() for s in line.split(',')] if i])
            elset_nodes.append(elem_line)
        elif f'*Elset, elset={setname.lower()}' in line:
            gather = True
            if 'generate' in line: generate = True
    return np.array(np.concatenate(elset_nodes))

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

        # Get bc nodes
        bcs_bot = get_nset_nodes(lines, 'bottom')
        bcs_top = get_nset_nodes(lines, 'top')
        out = job_dir.joinpath(f'{job_name}_bc_nodes_mfe.csv')
        pl.DataFrame(np.vstack([bcs_bot, bcs_top]), schema=['bottom', 'top'], orient='col').write_csv(out)

        # Get gage elements
        gage = get_elset_elems(lines, 'gage')
        out = job_dir.joinpath(f'{job_name}_gage_elems_mfe.csv')
        pl.DataFrame(gage, schema=['gage']).write_csv(out)

if __name__ == "__main__":
    main()