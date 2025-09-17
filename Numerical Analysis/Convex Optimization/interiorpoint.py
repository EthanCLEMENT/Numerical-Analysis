import numpy as np

def f_mu(x,mu):
    return 0.5 * x.T @ Q @ x + c.T @ x - mu * np.sum(np.log(x))

def grad_f_mu(x,mu):
    return Q @ x + c - mu * (1 / x)

def hess_f_mu(x,mu):
    return Q + mu * np.diag(1 / x**2)

def interior_point_qp(Q, c, A, b, x0, mu0=1.0, mu_tol=1e-6, tau=0.1, max_inner=20):
    x = x0
    mu = mu0
    lam = np.zeros(A.shape[0])
    history = []

    while mu > mu_tol:
        for _ in range(max_inner):
            g = grad_f_mu(x, mu)
            H = hess_f_mu(x, mu)
            J = A
            h_val = A @ x - b

            KKT = np.block([
                [H, J.T],
                [J, np.zeros((J.shape[0], J.shape[0]))]
            ])
            rhs = -np.concatenate([g + J.T @ lam, h_val])

            sol = np.linalg.solve(KKT, rhs)
            dx = sol[:len(x)]
            dlam = sol[len(x):]

            alpha = 1.0
            while np.any(x + alpha * dx <= 0):
                alpha *= 0.5

            x += alpha * dx
            lam += alpha * dlam

            if np.linalg.norm(dx) < 1e-6 and np.linalg.norm(h_val) < 1e-6:
                break

        history.append(f_mu(x, mu))
        mu *= tau 

    return x, lam, history



Q = np.array([[2, 0], [0, 2]])
c = np.array([-2, -5])
A = np.array([[1, 1]])
b = np.array([1])

x0 = np.array([0.5, 0.5]) 
x_star, lam_star, f_vals = interior_point_qp(Q, c, A, b, x0)
print("Optimal x:", x_star)
print("Optimal lambda:", lam_star)

