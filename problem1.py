import numpy as np
import matplotlib.pyplot as plt

m = 501
v = 90
eta = 3
dt = .5
alpha = .6
sigma = .5
ssigma = 1.5

P = np.array([[16,1,1,1,1],
              [1,16,1,1,1],
              [1,1,16,1,1],
              [1,1,1,16,1],
              [1,1,1,1,16]]) / 20

Phi = np.block([[np.array([[1, dt, (dt**2)/2],
                            [0, 1, dt],
                            [0, 0, alpha]]), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.array([[1, dt, (dt**2)/2],
                                               [0, 1, dt],
                                               [0, 0, alpha]])]])

Psi_Z = np.block([[np.array([(dt**2)/2, dt, 0]).reshape(-1, 1), np.zeros((3, 1))],
                   [np.zeros((3, 1)), np.array([(dt**2)/2, dt, 0]).reshape(-1, 1)]])

Psi_W = np.block([[np.array([(dt**2)/2, dt, 1]).reshape(-1, 1), np.zeros((3, 1))],
                   [np.zeros((3, 1)), np.array([(dt**2)/2, dt, 1]).reshape(-1, 1)]])

def simulate(P, m):
    # Define your simulation function here if not already defined
    pass

Z_index = simulate(P, m)
Z_statespace = np.array([[0, 0], [3.5, 0], [0, 3.5], [0, -3.5], [-3.5, 0]])
Z_simulated = np.zeros((2, m))
for n in range(m):
    Z_simulated[:, n] = Z_statespace[:, Z_index[n]]

X_simulated = np.zeros((6, m))
X_simulated[:, 0] = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]))

def newX(X, Z, dt, alpha, sigma):
    Phi_hat = np.array([[1, dt, dt**2/2],
                         [0, 1, dt],
                         [0, 0, alpha]])
    Phi = np.block([[Phi_hat, np.zeros((3, 3))],
                    [np.zeros((3, 3)), Phi_hat]])
    Psi_hat_z = np.array([dt**2/2, dt, 0])
    Psi_z = np.block([[Psi_hat_z.reshape(-1, 1), np.zeros((3, 1))],
                      [np.zeros((3, 1)), Psi_hat_z.reshape(-1, 1)]])
    Psi_hat_w = np.array([dt**2/2, dt, 1])
    Psi_w = np.block([[Psi_hat_w.reshape(-1, 1), np.zeros((3, 1))],
                      [np.zeros((3, 1)), Psi_hat_w.reshape(-1, 1)]])
    W = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2)).reshape(-1, 1)
    X_next = Phi @ X + Psi_z @ Z + Psi_w @ W
    return X_next

for n in range(m - 1):
    X_simulated[:, n + 1] = newX(X_simulated[:, n], Z_simulated[:, n], dt, alpha, sigma)

plt.figure()
plt.plot(X_simulated[0, :], X_simulated[3, :])
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()
