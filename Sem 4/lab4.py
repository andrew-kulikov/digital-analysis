import numpy as np
from cv2 import minMaxLoc


def check_stop(A):
    res = True
    rows, cols = A.shape
    for i in range(rows):
        for j in range(i, cols):
            if j != i:
                res = res & (abs(A[i, j]) < 10**(-2))
            if not res:
                break
    return res


def eigenvectors_jakobi(A, max_steps=10000):
    A_symm = np.copy(A)
    rows, cols = A_symm.shape
    U_last = np.identity(rows)
    for k in range(max_steps):
        if check_stop(A_symm):
            break
        _, max_val, _, max_loc = minMaxLoc(np.abs(np.tril(A_symm.T, -1).T))
        j, i = max_loc
        phi = np.pi / 4
        if i != j:
            phi = 0.5  * np.arctan(2 * A_symm[i, j] / (A_symm[i, i] - A_symm[j, j]))
        U = np.identity(rows)
        U[i, i] = np.cos(phi)
        U[i, j] = -np.sin(phi)
        U[j, i] = np.sin(phi)
        U[j, j] = np.cos(phi)
        #print('Phi:', phi)
        #print('U:', U)
        U_last = np.dot(U_last, U)
        A_symm = np.dot(U.T, np.dot(A_symm, U))
        print('A:', A_symm)
        print('==============')
    eigenvalues = [A_symm[i, i] for i in range(rows)]
    return eigenvalues, U_last


def main():
    A = np.array([[3.857, 0.239, 0.272, 0.258],
                  [0.491, 3.941, 0.131, 0.178],
                  [0.436, 0.281, 4.189, 0.416],
                  [0.317, 0.229, 0.326, 2.971]])
    A_symm = np.dot(A.T, A)
    eigenvalues, eigenvectors = eigenvectors_jakobi(A_symm)
    print('Eigenvalues:', eigenvalues)
    print('Eigenvectors:\n', eigenvectors)
    

if __name__ == '__main__':
    main()
	

Диапазон чисел, которые можно записать данным способом, зависит от количества бит, отведённых для представления мантиссы и показателя. На обычной 32-битной вычислительной машине, использующей двойную точность (64 бита), мантисса составляет 1 бит знак + 52 бита, показатель — 1 бит знак + 10 бит. Таким образом получаем диапазон точности примерно от 4,94·10−324 до 1.79·10308 (от 2−52 × 2−1022 до ~1 × 21024). В стандарте IEEE 754 несколько значений данного типа зарезервировано для обеспечения возможности представления специальных значений. К ним относятся значения NaN (Not a Number, «не число») и +/-INF (Infinity, бесконечность), получающихся в результате операций деления на ноль или при превышении числового диапазона. Также сюда попадают денормализованные числа, у которых мантисса меньше единицы. В специализированных устройствах (например, GPU) поддержка специальных чисел часто отсутствует. Существуют программные пакеты, в которых объём памяти выделенный под мантиссу и показатель задаётся программно, и ограничивается лишь объёмом доступной памяти ЭВМ (см. Арифметика произвольной точности).

