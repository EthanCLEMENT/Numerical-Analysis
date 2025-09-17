import numpy as np

def f(x):
    return np.log(1 + np.exp(-x[0])) + x[1]**2

def h(x):
    return x[0] + x[1] - 1

def grad_f(x):
    return np.array([
        -1 / (1 + np.exp(x[0])),
        2 * x[1]
    ])

def hess_f(x):
    d = np.exp(x[0]) / (1 + np.exp(x[0]))**2
    return np.array([
        [d, 0],
        [0, 2]
    ])

def jacobian_h(x):
    return np.array([[1,1]])

def newton_equality_constrained(f, grad_f, hess_f, h, jac_h, x0, lam0, max_iter=100, tol=1e-6):
    x = x0
    lam = lam0
    history = []

    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        J = jac_h(x)
        hx = h(x)

        KKT_matrix = np.block([
            [H, J.T],
            [J, np.zeros((1, 1))]
        ])
        RHS = -np.concatenate([g + J.T @ lam, [hx]])

        sol = np.linalg.solve(KKT_matrix, RHS)
        dx = sol[:2]
        dlam = sol[2:]

        x = x + dx
        lam = lam + dlam
        history.append(f(x))

        if np.linalg.norm(dx) < tol and abs(h(x)) < tol:
            break

    return x, lam, history


x0 = np.array([0.5, 0.5])
lam0 = np.array([0.0])

x_opt, lam_opt, hist = newton_equality_constrained(f, grad_f, hess_f, h, jacobian_h, x0, lam0)

print("Optimal x:", x_opt)
print("Optimal lambda:", lam_opt)
print("Constraint value:", h(x_opt))

