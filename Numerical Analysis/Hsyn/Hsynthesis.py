import numpy as np
from scipy.linalg import eig, eigvals

def solve_hinfinity_riccati(A, B1, B2, C1, C2, D12, D21, gamma):
    R = D12.T @ D12
    S = D21 @ D21.T

    # Riccati for X
    Ahat = A - B2 @ np.linalg.inv(R) @ D12.T @ C1
    Qhat = C1.T @ (np.eye(D12.shape[0]) - D12 @ np.linalg.inv(R) @ D12.T) @ C1
    HX = np.block([
        [Ahat, (1/gamma**2) * B1 @ B1.T - B2 @ np.linalg.inv(R) @ B2.T],
        [-Qhat, -Ahat.T]
    ])
    eigvalsX, eigvecsX = eig(HX)
    V = eigvecsX[:, eigvalsX.real < 0]
    Z1, Z2 = V[:A.shape[0], :], V[A.shape[0]:, :]
    X = Z2 @ np.linalg.inv(Z1)

    # Riccati for Y
    Afhat = A - B1 @ D21.T @ np.linalg.inv(S) @ C2
    Qfhat = B1 @ (np.eye(D21.shape[1]) - D21.T @ np.linalg.inv(S) @ D21) @ B1.T
    HY = np.block([
        [Afhat.T, (1/gamma**2)*C1.T@C1 - C2.T@np.linalg.inv(S)@C2],
        [-Qfhat, -Afhat]
    ])
    eigvalsY, eigvecsY = eig(HY)
    V = eigvecsY[:, eigvalsY.real < 0]
    Z1, Z2 = V[:A.shape[0], :], V[A.shape[0]:, :]
    Y = Z2 @ np.linalg.inv(Z1)

    return X.real, Y.real


def augmented(A, B1, B2, C1, C2, D11, D12, D21, D22):
    return np.block([[A, B1, B2],
                     [C1, D11, D12],
                     [C2, D21, D22]])


def bisection_algorithm(A, B1, B2, C1, C2, D12, D21, epsilon=1e-3):
    # Initial bounds
    gamma_low = 1e-5
    gamma_high = 1e3

    X = None
    Y = None

    while gamma_high - gamma_low > epsilon:
        gamma = 0.5 * (gamma_low + gamma_high)

        try:
            X, Y = solve_hinfinity_riccati(A, B1, B2, C1, C2, D12, D21, gamma)

            # Coupling condition
            rho = max(abs(eigvals(X @ Y)))

            if rho < gamma**2:
                gamma_high = gamma
            else:
                gamma_low = gamma

        except Exception:
            gamma_low = gamma

    return gamma_high, X, Y


def hinfinity_controller(gamma, A, B1, B2, C1, C2, D11, D12, D21, D22, X, Y):
    R = D12.T @ D12
    S = D21 @ D21.T

    # Gains
    F = np.linalg.inv(R) @ (B2.T @ X + D12.T @ C1)
    L = (Y @ C2.T + B1 @ D21.T) @ np.linalg.inv(S)

    # Controller realization
    AK = A - B2 @ F - L @ C2 + (1/gamma**2) * B1 @ B1.T @ X
    BK = L
    CK = -F
    DK = np.zeros((D12.shape[1], D21.shape[0]))

    return AK, BK, CK, DK


A = np.array([[0, 1],
              [-2, -3]])
B1 = np.array([[0],
               [1]])
B2 = np.array([[0],
               [1]])
C1 = np.array([[1, 0]])   
C2 = np.array([[0, 1]])   
D11 = np.array([[0]])
D12 = np.array([[1]])
D21 = np.array([[1]])
D22 = np.array([[0]])

print("Augmented plant:\n", augmented(A, B1, B2, C1, C2, D11, D12, D21, D22))

gamma_star, X, Y = bisection_algorithm(A, B1, B2, C1, C2, D12, D21)
print("Optimal gamma:", gamma_star)

AK, BK, CK, DK = hinfinity_controller(gamma_star, A, B1, B2, C1, C2, D11, D12, D21, D22, X, Y)
print("Controller matrices:")
print("A_K =\n", AK)
print("B_K =\n", BK)
print("C_K =\n", CK)
print("D_K =\n", DK)


