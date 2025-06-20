import numpy as np

def solve_kkt(Q,c,A,b):

    n = Q.shape[0]
    m = A.shape[0]
    c = c.reshape(-1, 1)
    b = b.reshape(-1, 1)

    top = np.hstack((Q, A.T))

    bottom = np.hstack((A, np.zeros((m, m))))

    KKT = np.vstack((top, bottom))

    RHS = np.vstack((-c,b))

    solution = np.linalg.solve(KKT,RHS)
    x = solution[:n].ravel()
    lam = solution[n:].ravel()

    return x,lam

Q = np.array([[2, 0], [0, 2]])
c = np.array([-2, -5])
A = np.array([[1, 1]])
b = np.array([1])

x, lam = solve_kkt(Q, c, A, b)
print("x =", x.ravel())
print("lambda =", lam.ravel())

print("Ax =", A @ x)              
print("KKT residual:", Q @ x + c + A.T @ lam)  