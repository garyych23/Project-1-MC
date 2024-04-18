import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Load data
pos_vec_file = open('stations.txt', "r")
Y_file = open('RSSI-measurements.txt', "r")

# Parameters
m = 501
dt = 0.5
alpha = 0.6
sigma = 0.5
N = 20000
v = 90
eta = 3
zeta = 1.5

Y = np.zeros((6,m))
c_l = 0
for l in Y_file:
    line = l.split(",")
    for k in range(m):
        Y[c_l,k] = float(line[k])
    c_l += 1

pos_vec = np.zeros((2,6))
c_l = 0
for l in pos_vec_file:
    line = l.split(",")
    for k in range(6):
        pos_vec[c_l,k] = float(line[k])
    c_l += 1
    
print(Y.shape)

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

Z = np.array([[0, 0], [3.5, 0], [0, 3.5], [0, -3.5], [-3.5, 0]])

# Sequential importance sampling
tau = np.zeros((2, m))  # vector of estimates

def findDist(l, x, pos_vec):
    p = np.array([x[0], x[3]]) - pos_vec[l - 1]
    dist = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    return dist

def p(x, y):
    distances = np.array([v - 10 * eta * np.log10(findDist(1, x, pos_vec)),
                          v - 10 * eta * np.log10(findDist(2, x, pos_vec)),
                          v - 10 * eta * np.log10(findDist(3, x, pos_vec)),
                          v - 10 * eta * np.log10(findDist(4, x, pos_vec)),
                          v - 10 * eta * np.log10(findDist(5, x, pos_vec)),
                          v - 10 * eta * np.log10(findDist(6, x, pos_vec))])
    return scipy.stats.multivariate_normal.pdf(y, distances.T, cov=np.eye(6)*sigma**2)


part = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]), size=N).T
weights = p(part, Y[:, 0])
tau[:, 0] = np.sum(part[::3, :] * weights, axis=1) / np.sum(weights)
tau[:, 1] = np.sum(part[3::3, :] * weights, axis=1) / np.sum(weights)

mc = scipy.stats.rv_discrete(values=(range(5), P.ravel()))
simulate_Z = mc.rvs(size=m)

for k in range(m - 1):
    part = Phi @ part + np.tile(Psi_Z @ Z[:, simulate_Z[k]], (1, N)) + Psi_W @ np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag([0.25, 0.25]), size=N).T
    weights = p(part, Y[:, k + 1])
    tau[:, k + 1] = np.sum(part[::3, :] * weights, axis=1) / np.sum(weights)
    tau[:, k + 1] = np.sum(part[3::3, :] * weights, axis=1) / np.sum(weights)
    print(np.max(np.abs(weights)))

plt.figure()
plt.plot(tau[0, :], tau[1, :], '*')
plt.plot(pos_vec[:, 0], pos_vec[:, 1], '*', color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
