import numpy as np


def QR(A):
    m, n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((m,n))

    r11 = np.linalg.norm(A[:,0])
    q1 = A[:,0] / r11

    Q[:, 0] = q1
    R[:, 0] = r11
    for j in range(1, n):
        xj = A[:,j]
        qhat = xj
        for i in range(j):
            qi = Q[:, i]
            rij = np.inner(qhat, qi)
            R[i, j] = np.inner(qi, qhat)
            qhat -= rij * qi
        rjj = np.linalg.norm(qhat)
        qj = qhat / rjj
        Q[:, j] = qj
        R[j,j] = rjj

    return Q, R 


np.random.seed(2)
A = np.random.rand(3, 3)
print(A)
print(QR(A))
print(np.linalg.qr(A))