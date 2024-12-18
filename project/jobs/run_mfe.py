import argparse
import pathlib
import re
import time
from datetime import datetime

import polars as pl
import numpy as np

import mfe.baseclasses
import mfe.solve
import mfe.utils
import mfe.load
import mfe.plot
import mfe.elem_lib

JOB_DIR = pathlib.Path('linear')
D = mfe.utils.D_transversely_isotropic_plane_stress(E11=231000, E22=15000, nu12=0.21, G12=15800)
THICKNESS = 0.001
APPLIED_DISP = 0.1
PLOT_ELEM_GRID = mfe.utils.make_natural_grid(4)
CONVERGED_MESH = 1

def _argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='sensitivity', nargs='?')
    return parser.parse_args()

def load_mesh(job_dir: pathlib.Path, job_name: str) -> tuple[np.ndarray]:
    # Set the current mesh files
    mesh_files = {
        'connectivity': job_dir.joinpath(f'{job_name}_connectivity_mfe.csv'),
        'node_coords': job_dir.joinpath(f'{job_name}_nodes_mfe.csv'),
    }

    # Get connectivity matrix and nodal coordinates, then assemble the mesh
    G, node_coords = mfe.utils.read_mesh_from_csv(**mesh_files)
    elems = mfe.solve.assemble_mesh(G, node_coords)

    return G, elems

def load_bcs(job_dir: pathlib.Path, job_name: str) -> dict[str, dict[int, float]]:
    # Load the boundary condition nodes
    bc_file = job_dir.joinpath(f'{job_name}_bc_nodes_mfe.csv')
    bcs = pl.read_csv(bc_file)

    # Set the boundary conditions
    disp_bcs = {'y_disp': {}, 'x_disp': {}}
    for node in bcs.select('bottom').to_series():
        disp_bcs['x_disp'].update({node: 0})
        disp_bcs['y_disp'].update({node: 0})

    for node in bcs.select('top').to_series():
        disp_bcs['x_disp'].update({node: 0})
        disp_bcs['y_disp'].update({node: APPLIED_DISP})
    return disp_bcs

def set_elem_params(job_name: str, elems: np.ndarray[mfe.baseclasses.Element2D]) -> np.ndarray[mfe.baseclasses.Element2D]:
    # Set the element material property matrix, thickness, and orientation
    theta = re.search(r'_(\d+)', job_name).group(1)
    T = mfe.utils.make_transform_matrix_2D(int(theta))
    thickness = 0.001
    for e in elems:
        e.D = D.copy()
        e.thickness = thickness
        e.T = T.copy()
    return elems

def run_simulation(G: np.ndarray, elems: np.ndarray, disp_bcs: dict[str, dict[int, float]], K_sensitivity: float = 1.0) -> tuple[np.ndarray]:
    # Assemble solution matrices
    loads = [[] for _ in elems]
    K, F = mfe.solve.assemble_global_solution(G, elems, loads, 2)

    # Apply disp BCs
    K, F = mfe.solve.apply_disp_bcs(disp_bcs['x_disp'], disp_bcs['y_disp'], K, F)

    # Apply sensititity to K for calc
    K = K*(1 + K_sensitivity)

    # Solve
    Q = np.matmul(np.linalg.inv(K), F)

    return Q, F

def run_sensitivity_study() -> None:
    results = {0: [], 90: [], 45: []}
    job_dirs = []
    for fp in JOB_DIR.glob(f'*mesh{CONVERGED_MESH}*'):
        if not fp.is_dir: continue
        if not any(f'_{theta}' in fp.as_posix() for theta in results.keys()): continue
        job_dirs.append(fp)

    results = []
    for ks in np.linspace(-0.01, 0.01, 2):
        char = {'p': ks}
        for fp in JOB_DIR.glob(f'*mesh*'):
            job_name = fp.name
            theta = re.search(r'_(\d+)', job_name).group(1)
            if not any(f'_{theta}' in fp.as_posix() for theta in [0, 90, 45]): continue
            # char.update({'theta': int(theta)})
            print(f'\n*starting job -> {job_name}')

            # Load the mesh connectivity and elements
            print('loading mesh from file...')
            G, elems = load_mesh(fp, job_name)
            char.update({
                'elements': len(elems),
            })

            # Load the boundary conditions
            print('loading bcs from file...')
            disp_bcs = load_bcs(fp, job_name)

            # Get gage elements
            gage_elem_labels = pl.read_csv(fp.joinpath(f'{job_name}_gage_elems_mfe.csv'))
            gage_elems = np.array(elems)[gage_elem_labels.to_series().to_list()]
            gage_G = G[gage_elem_labels.to_series().to_list()]

            # Set element parameters (thickness, rotation, material stiffness)
            print('initializing element parameters...')
            elems = set_elem_params(job_name, elems)

            # Run the analysis
            start = time.perf_counter()
            print('running simulation')
            Q, F = run_simulation(G, elems, disp_bcs, K_sensitivity=ks)
            print('simulation complete!')
            end = time.perf_counter()

            # Gage quantities
            # sig, eps = mfe.solve.map_stress_strain_to_assembly(gage_G, gage_elems, Q, PLOT_ELEM_GRID, loc_sys=True)
            sig, eps = mfe.solve.map_stress_strain_to_assembly(gage_G, gage_elems, Q, PLOT_ELEM_GRID, loc_sys=False)
            sig, eps = [np.mean(field, axis=(0, 1)) for field in [sig, eps]]

            # Results
            if theta == '0':
                char.update({'E22': sig[1][0]/eps[1][0]})
            elif theta == '90':
                char.update({'E11': sig[1][0]/eps[1][0]})
                char.update({'nu12': -eps[0][0]/eps[1][0]})
            elif theta == '45':
                char.update({'G12': sig[1][0]/(2*(eps[1][0] - eps[0][0]))})
            results.append(pl.DataFrame(char))
            # print(char)
            # input()
    df = pl.concat(results, how='diagonal').group_by('elements', 'p').agg(pl.all().drop_nulls().first())
    df = df.with_columns(
        (100*abs(pl.col('E11') - 231000)/pl.col('E11')).name.suffix('_tol'),
        (100*abs(pl.col('E22') - 15000)/pl.col('E22')).name.suffix('_tol'),
        (100*abs(pl.col('nu12') - 0.21)/pl.col('nu12')).name.suffix('_tol'),
        (100*abs(pl.col('G12') - 15800)/pl.col('G12')).name.suffix('_tol'),
    )
    print('study complete, writing results to file...')
    df.write_csv('sensitivity_study_results.csv')

def run_convergence_study() -> None:
    for fp in JOB_DIR.glob('*'):
        if not fp.is_dir(): continue
        job_name = fp.name
        print(f'\n*starting job -> {job_name}')

        # Create summary
        summary = {
            'Job name': job_name,
            'Date': datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            'Work dir': fp
        }

        # Load the mesh connectivity and elements
        print('loading mesh from file...')
        G, elems = load_mesh(fp, job_name)
        summary.update({
            'Elements': len(elems),
            'Nodes': np.max(G)
        })

        # Load the boundary conditions
        print('loading bcs from file...')
        disp_bcs = load_bcs(fp, job_name)

        # Get gage elements
        gage_elem_labels = pl.read_csv(fp.joinpath(f'{job_name}_gage_elems_mfe.csv'))
        gage_elems = np.array(elems)[gage_elem_labels.to_series().to_list()]
        gage_G = G[gage_elem_labels.to_series().to_list()]

        # Set element parameters (thickness, rotation, material stiffness)
        print('initializing element parameters...')
        elems = set_elem_params(job_name, elems)

        # Run the analysis
        start = time.perf_counter()
        print('running simulation')
        Q, F = run_simulation(G, elems, disp_bcs)
        print('simulation complete!')
        end = time.perf_counter()

        # Map displacements, stresses, and strains to the assembly
        print('writing results...')

        # Nodal quantities
        np.save(fp.joinpath('mesh'), np.vstack([elem.x_global for elem in elems]))
        np.save(fp.joinpath('nodal_disp'), Q)
        np.save(fp.joinpath('nodal_reaction_force'), F)

        # Element quantities
        x_assembly = mfe.solve.build_assembly_coord_grid(G, elems, PLOT_ELEM_GRID)
        np.save(fp.joinpath('assembly_grid'), x_assembly)
        Q_assembly = mfe.solve.map_nodal_field_to_assembly(G, elems, Q, PLOT_ELEM_GRID)
        np.save(fp.joinpath('assembly_disp'), Q_assembly)
        F_assembly = mfe.solve.map_nodal_field_to_assembly(G, elems, F, PLOT_ELEM_GRID)
        np.save(fp.joinpath('assembly_reaction_force'), F_assembly)
        sig, eps = mfe.solve.map_stress_strain_to_assembly(G, elems, Q, PLOT_ELEM_GRID, loc_sys=True)
        np.save(fp.joinpath('assembly_stress'), sig)
        np.save(fp.joinpath('assembly_strain'), eps)
        sig, eps = mfe.solve.map_stress_strain_to_assembly(G, elems, Q, PLOT_ELEM_GRID, loc_sys=False)
        np.save(fp.joinpath('assembly_stress_glob'), sig)
        np.save(fp.joinpath('assembly_strain_glob'), eps)

        # Gage quantities
        x_gage = mfe.solve.build_assembly_coord_grid(gage_G, gage_elems, PLOT_ELEM_GRID)
        np.save(fp.joinpath('gage_grid'), x_gage)
        Q_gage = mfe.solve.map_nodal_field_to_assembly(gage_G, gage_elems, Q, PLOT_ELEM_GRID)
        np.save(fp.joinpath('gage_disp'), Q_gage)
        F_gage = mfe.solve.map_nodal_field_to_assembly(gage_G, gage_elems, F, PLOT_ELEM_GRID)
        np.save(fp.joinpath('gage_reaction_force'), F_gage)
        sig, eps = mfe.solve.map_stress_strain_to_assembly(gage_G, gage_elems, Q, PLOT_ELEM_GRID, loc_sys=True)
        np.save(fp.joinpath('gage_stress'), sig)
        np.save(fp.joinpath('gage_strain'), eps)
        sig, eps = mfe.solve.map_stress_strain_to_assembly(gage_G, gage_elems, Q, PLOT_ELEM_GRID, loc_sys=False)
        np.save(fp.joinpath('gage_stress_glob'), sig)
        np.save(fp.joinpath('gage_strain_glob'), eps)

        # Update summary
        summary.update({
            'Simulation time': f'{end - start:.3f} seconds',
            'status': 'COMPLETE'
        })

        # Write job summary to file
        with open(fp.joinpath('job_summary.log'), 'w+') as f:
            for k, v in summary.items():
                f.write(f'{k}: {v}\n')

def main() -> None:
    funcs = {
        'sensitivity': run_sensitivity_study,
        'convergence': run_convergence_study,
    }
    args = _argparse()
    funcs[args.mode]()    
        
if __name__ == "__main__":
    main()