import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(theta, Y):
    y1, y2 = Y
    dy1 = y2
    dy2 = np.sin(theta) - 2*y2 - 3*y1
    return np.array([dy1, dy2])

def ode(f, theta0, y0, h, n):
    theta_vals = [theta0]
    y_vals = [y0]

    theta = theta0
    y = y0

    for i in range(n):
        k1 = h * f(theta,y)
        k2 = h * f(theta + h / 2, y + k1 / 2)
        k3 = h * f(theta + h / 2, y + k2 / 2)
        k4 = h * f(theta + h, y + k3)

        y += (k1 + 2*k2 + 2*k3 + k4) / 6

        theta = theta + h
        theta_vals.append(theta)
        y_vals.append(y.copy())

    return np.array(theta_vals), np.array(y_vals)

y0 = np.array([0.0,0.0]) # y, y'
theta0 = 0 # theta
n = 100
h = 0.1

theta, Y = ode(f, theta0, y0, h, n)
theta_span = (theta0, theta0 + n*h)
sol = solve_ivp(f, theta_span, y0, method='RK45', t_eval=theta)

# Plot results
plt.figure(figsize=(10,6))

plt.plot(theta, Y[:,0], 'o-', label="RK4 (y)")
plt.plot(theta, Y[:,1], 'o-', label="RK4 (y')")
plt.plot(sol.t, sol.y[0], 'k--', label="solve_ivp (y)")
plt.plot(sol.t, sol.y[1], 'r--', label="solve_ivp (y')")

plt.xlabel(r"$\theta$")
plt.ylabel("Solution")
plt.title("Comparison of RK4 vs SciPy solve_ivp")
plt.legend()
plt.grid(True)
plt.show()
