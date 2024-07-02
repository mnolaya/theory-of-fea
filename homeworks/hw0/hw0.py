import numpy as np

# Vectors and matrices to be used for solving problems
a_m = np.array([[3, 5, 6], [7, 1, 9], [0, 4, 6]])
b_m = np.array([[1, 0, 1], [4, 2, 0], [2, 0, 0]])
a_v = np.array([[1], [2], [0]])
b_v = np.array([[3], [97.95], [2]])

# Transpose a vector or matrix
def transpose(arr: np.ndarray) -> np.ndarray:
    temp = []  # Temp list for creating tranposed array
    # Loop through indices in tranposed order (i, j) -> (j, i)
    for j in range(arr.shape[1]): 
        row = []
        for i in range(arr.shape[0]):
            row.append(arr[i, j])
        temp.append(row)
    transposed = np.array(temp)

    # Validate manually transposed array with numpy's built-in
    if np.all(transposed == arr.T):
        return transposed
    else:
        print('error! manual transpose does not match result of numpy built-in')
        print(f'manual: {transposed}')
        print(f'numpy: {arr.T}')

# Perform matrix multiplication for two matrices a[m, n] and b[p, q] where n == p
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    # Create an array of zeros for the resulting matrix multiplication
    c = np.zeros(shape=(a.shape[0], b.shape[1]))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            # Multiply & sum row of [a] by column of [b] for each i, j
            for k, l in zip(range(a.shape[1]), range(b.shape[0])):
                c[i, j] += a[i, k]*b[l, j]

    # Validate manual multiplication with numpy's built-in
    if np.all(c == np.matmul(a, b)):
        if c.shape == (1, 1): c = c[0, 0]
        return c
    else:
        print('error! manual matrix multiplication does not match result of numpy built-in')
        print(f'manual: {c}')
        print(f'numpy: {np.matmul(a, b)}')

# Print formatted result for homework problem
def print_result(problem: str, res: np.ndarray) -> None:
    print(f'---')
    print(f'Problem {problem}')
    print(f'{res}\n')

def main() -> None:
    # 1.a.
    res = matmul(transpose(a_v), b_v)
    print_result('1.a', res)
    
    # 1.b.
    res = matmul(a_m, a_v)
    print_result('1.b', res)

    # 1.c.
    res = matmul(transpose(a_v), matmul(a_m, b_v))
    print_result('1.c', res)

    # 1.d.
    res = matmul(a_m, b_m)
    print_result('1.d', res)

    # 1.e.
    res = matmul(a_m, transpose(b_m))
    print_result('1.e', res)
    

if __name__ == '__main__':
    main()