import pathlib

import subprocess

JOB_DIR = pathlib.Path('linear')

for fp in JOB_DIR.glob('*mesh5*'):
    if not fp.is_dir(): continue
    job_name = fp.stem
    inp = job_name + '.inp'
    subprocess.call(['abaqus', 'inter', '-j', inp, '-ask_delete', 'off', '-cpus', '4'], shell=True, cwd=fp)
