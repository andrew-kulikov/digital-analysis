import numpy as np
import lab1


def prepare_system(A, b):
    rows, cols = A.shape
    a = A.copy()
    b1 = b.copy()
    for i in range(rows):
        b1[i] /= a[i][i]
        a[i, :] = -a[i, :] / a[i][i]
        a[i][i] = 0       
    return a, b1


def solve_iterations(a, b, e, start_x, max_iterations=10000):
    cur_x = start_x.copy()
    prev_x = start_x.copy()
    final_iteration = max_iterations
    for i in range(max_iterations):
        cur_x = np.dot(a, cur_x) + b
        #print(cur_x, lab1.vector_norm(cur_x - prev_x))
        #print('############')
        if lab1.vector_norm(cur_x - prev_x) < e:
            final_iteration = i
            break
        prev_x = cur_x.copy()
    return cur_x, final_iteration + 1


def solve_zeidel(a, b, e, start_x, max_iterations=10000):
    cur_x = start_x.copy()
    prev_x = start_x.copy()
    final_iteration = max_iterations
    n = a.shape[0]
    
    def get_LU(a):
        L = np.zeros(a.shape)
        U = np.zeros(a.shape)
        rows = a.shape[0]
        for i in range(rows):
            L[i, :i] = a[i, :i]
            U[i, i:] = a[i, i:]
        return L, U
    
    L, U = get_LU(a) 
    E = np.identity(n)
    for i in range(max_iterations):
        cur_x = np.dot(
                    np.dot(
                        np.linalg.inv(E - L), U
                        ), cur_x
                    ) + np.dot(np.linalg.inv(E - L), b)
        print(cur_x, lab1.vector_norm(cur_x - prev_x))
        if lab1.vector_norm(cur_x - prev_x) < e:
            final_iteration = i
            break
        prev_x = cur_x.copy()
    return cur_x, final_iteration + 1

    
def main():
    A = np.array([[3.857, 0.239, 0.272, 0.258],
                  [0.491, 3.941, 0.131, 0.178],
                  [0.436, 0.281, 4.189, 0.416],
                  [0.317, 0.229, 0.326, 2.971]])
    b = np.array([0.19, 0.179, 0.753, 0.860])
    e = 0.01
    a, beta = prepare_system(A, b)
    start_x = np.zeros(len(beta))
    x, iterations = solve_iterations(a, beta, e, start_x)
    #x1 = lab1.solve_gaussian(A, b)
    x1, iterations1 = solve_zeidel(a, beta, e, start_x)
    print(x, iterations)
    print(x1, iterations1)

if __name__ == '__main__':
    main()