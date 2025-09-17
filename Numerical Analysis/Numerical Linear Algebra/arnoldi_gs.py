import numpy as np
from scipy.sparse import kron, eye, diags

def arnoldi_cgs(A, v, m):
    n = A.shape[0]
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))

    V[:, 0] = v / np.linalg.norm(v)

    for j in range(m):
        w = A @ V[:, j]
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i], w)
        w = A @ V[:, j] - V[:, :j+1] @ H[:j+1, j]
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] == 0:
            return V, H
        V[:, j+1] = w / H[j+1, j]

    return V, H

def arnoldi_mgs(A, v, m):
    n = A.shape[0]
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m))

    V[:, 0] = v / np.linalg.norm(v)

    for j in range(m):
        w = A @ V[:, j]
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] == 0:
            return V, H
        V[:, j+1] = w / H[j+1, j]

    return V, H

def check_relations(A, V, H, m):
    Vm = V[:, :m]
    Hm = H[:m+1, :m]
    ortho_err = np.linalg.norm(Vm.T @ Vm - np.eye(m))
    arnoldi_err = np.linalg.norm(Vm.T @ (A @ Vm) - Hm[:m, :])
    return ortho_err, arnoldi_err

def poisson_2d(N):
    e = np.ones(N)
    T = diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N), format="csr")
    I = eye(N, format="csr")
    A = kron(I, T) + kron(T, I)
    return A

N = 10
A = poisson_2d(N)
n = A.shape[0]

e1 = np.zeros(n)
e1[0] = 1.0

for m in [10, 20, 30, 40, 50]:
    V_cgs, H_cgs = arnoldi_cgs(A, e1, m)
    V_mgs, H_mgs = arnoldi_mgs(A, e1, m)

    ortho_cgs, arnoldi_cgs_err = check_relations(A, V_cgs, H_cgs, m)
    ortho_mgs, arnoldi_mgs_err = check_relations(A, V_mgs, H_mgs, m)

    print(f"\nm = {m}")
    print(f"  CGS: ||V^T V - I|| = {ortho_cgs:.2e}, ||V^T A V - H|| = {arnoldi_cgs_err:.2e}")
    print(f"  MGS: ||V^T V - I|| = {ortho_mgs:.2e}, ||V^T A V - H|| = {arnoldi_mgs_err:.2e}")

def check_skew(H, m):
    Hm = H[:m, :m]  
    skew_err = np.linalg.norm(Hm + Hm.T)
    return skew_err

def antisymmetric_1d(N):
    e = np.ones(N)
    return diags([-e, e], [-1, 1], shape=(N, N), format="csr")

N = 10
A = antisymmetric_1d(N)
n = A.shape[0]
e1 = np.zeros(n); e1[0] = 1.0

for m in [5, 10]:
    V, H = arnoldi_mgs(A, e1, m)
    ortho_err, arnoldi_err = check_relations(A, V, H, m)
    skew_err = check_skew(H, m)
    print(H)
    print(f"\nm = {m}")
    print(f"  ||V^T V - I|| = {ortho_err:.2e}")
    print(f"  ||V^T A V - H|| = {arnoldi_err:.2e}")
    print(f"  Skew-symmetry check: ||H + H^T|| = {skew_err:.2e}")

