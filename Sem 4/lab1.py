import numpy as np


def get_minor(A, i, j):
    A = np.column_stack((A[:, :j], A[:, j+1:]))
    A = np.row_stack((A[:i, :], A[i+1:, :]))
    return np.linalg.det(A)


def inversed_matr(A):
    rows, cols = A.shape
    B = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            minor = get_minor(A, i, j)
            B[i][j] = (-1)**(i + j) * minor
    return B.T / np.linalg.det(A)


def solve_gaussian(A, b1):
    rows, cols = A.shape
    b = b1.copy()
    
    def diagonalize(A, b):
        B = np.copy(A)
        for j in range(cols - 1):
            for i in range(j + 1, rows):
                if B[j][j] != 0:
                    row_mul = B[i][j] / B[j][j]
                else:
                    row_mul = 0
                for k in range(j, rows):
                    B[i][k] -= B[j][k] * row_mul
                b[i] -= b[j] * row_mul
        return B, b
    
    B, b = diagonalize(A, b)
    x = np.zeros(rows)
    for i in range(rows - 1, -1, -1):
        x[i] = b[i] / B[i][i]
        for k in range(i):
            b[k] -= B[k][i] * x[i]
    return x


def matrix_norm(A):
    A1 = np.abs(A)
    return np.max([sum(A1[i, :]) for i in range(A1.shape[0])])


def vector_norm(v):
    return np.max(np.abs(v))
    

def main():
    A = np.array([[3.857, 0.239, 0.272, 0.258],
                  [0.491, 3.941, 0.131, 0.178],
                  [0.436, 0.281, 4.189, 0.416],
                  [0.317, 0.229, 0.326, 2.971]])
    b = np.array([0.19, 0.179, 0.753, 0.860])
    dx = 0.001
    db = 0.001
    x = solve_gaussian(A, b)
    print('Solution: ', x)
    dx_rel = dx / vector_norm(x)
    db_rel = db / vector_norm(b)
    dsyst = matrix_norm(inversed_matr(A)) * db
    dsyst_rel = matrix_norm(A) * matrix_norm(inversed_matr(A)) * db_rel
    print('Error: {err}, Relative error: {rel_err}'.format(err=dsyst, rel_err=dsyst_rel))
    
  
if __name__ == '__main__':
    main()