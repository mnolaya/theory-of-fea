import subprocess

for theta in [0, 30, 45, 60, 90]:
    inp = f'proj_{theta}.inp'
    subprocess.call(['abaqus', 'inter', '-j', inp, '-ask_delete', 'off', '-cpus', '4'], shell=True)
