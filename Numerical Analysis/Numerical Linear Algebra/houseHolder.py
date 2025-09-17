import numpy as np

import numpy as np

def houseHolder(A):
    A = A.copy().astype(float)
    m, n = A.shape
    Q = np.eye(m)

    for k in range(n):
        x = A[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
        v = v / np.linalg.norm(v)

        H_k = np.eye(m)
        H_k[k:, k:] -= 2.0 * np.outer(v, v)

        A = H_k @ A
        Q = Q @ H_k

    R = A
    return Q, R



np.random.seed(2)
A = np.random.rand(3, 3)
print(A)
print(houseHolder(A))
print(np.linalg.qr(A))