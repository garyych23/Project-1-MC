import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(23)

# Parameters
m = 600
dt = 0.5
alpha = 0.6
sigma = 0.5

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


# Evolution of the Markov chain Z
def markov_chain(m,P,x0):
    X=np.zeros(m,dtype=np.int64) #on indique que X contiendra seulement des entiers.
    X[0]=x0
    for k in range(m-1):
        X[k+1] = np.random.choice(a=range(len(P)), p=P[X[k],:])
    return X

z0 = random.randint(0, 4)
Z_index = markov_chain(m,P,z0)
Z_state= np.array([[0, 0], [3.5, 0], [0, 3.5], [0, -3.5], [-3.5, 0]])
Z_sim = np.zeros((2, m))
for n in range(m):
    Z_sim[:, n] = Z_state[Z_index[n], :]
    
# Evolution of X
X_sim = np.zeros((6, m))
X_sim[:, 0] = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]))

for n in range(m-1):
    W = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2)).reshape(-1, 1)
    X_sim[:,n+1] = Phi @ X_sim[:,n] + Psi_Z @ Z_sim[:,n] + Psi_W @ W   

plt.figure()
plt.plot(X_sim[0, :], X_sim[3, :])
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()
