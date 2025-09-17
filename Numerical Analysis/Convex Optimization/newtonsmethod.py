import numpy as np

def newton(Q, c, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    Minimize f(x) = 1/2 x^T Q x + c^T x using gradient descent.

    Parameters:
        Q (ndarray): Positive definite matrix (n x n)
        c (ndarray): Vector (n,)
        x0 (ndarray): Initial guess (n,)
        alpha (float): Step size
        max_iter (int): Maximum number of iterations
        tol (float): Stopping threshold for gradient norm

    Returns:
        x (ndarray): Final estimate
        history (list): Function value at each iteration
    """
    x = x0
    history = []
    beta = 0.5 
    tau = 0.5

    for i in range(max_iter):
        alpha_k = alpha 
        gradient = Q @ x + c
        gradient_norm = np.linalg.norm(gradient)

        hessian = Q

        f = 0.5*x.T @ Q @ x + c.T @ x
        history.append(f)

        if gradient_norm < tol:
            break

        p = -np.linalg.solve(hessian, gradient)
        
        while True: 
            x_new = x + alpha_k * p
            f_new = 0.5 * x_new.T @ Q @ x_new + c.T @ x_new
            if f_new <= f - beta * alpha_k * gradient_norm**2:
                break
            alpha_k *= tau 

        x = x_new


    return x, history

Q = np.array([[1000, 0], [0, 1]])
c = np.array([-1, -2])

x0 = np.array([0.0, 0.0])

x_opt, f_vals = newton(Q, c, x0, alpha=0.1)

print("Optimal x:", x_opt)


