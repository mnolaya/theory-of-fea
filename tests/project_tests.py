import numpy as np

def rot_tens(theta: float) -> np.ndarray:
    theta_r = np.deg2rad(theta)
    return np.array(
        [[np.cos(theta_r), -1.0*np.sin(theta_r)],
        [1.0*np.sin(theta_r), np.cos(theta_r)]]
    )

def rot_r2(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.matmul(R, np.matmul(A, R.T))

rot = rot_tens(0)
eps = np.array([[0, 0], [0, 5]])
print(rot)
print(eps)
print(rot_r2(eps, rot))

for deg in [0, 45, 60, 90]:
    print(f'theta = {deg}')
    rot = rot_tens(deg)
    print(rot)
    print(rot_r2(eps, rot))