import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import random

random.seed(23)

# Load data
pos_vec_file = open('stations.txt', "r")
Y_file = open('RSSI-measurements-unknown-sigma.txt', "r")

# Parameters
m = 501
dt = 0.5
alpha = 0.6
sigma = 0.5
N = 10000
v = 90
eta = 3

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

Z_state= np.array([[0, 0], [3.5, 0], [0, 3.5], [0, -3.5], [-3.5, 0]])


# load and read data
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

############### Sequential importance sampling ####################
def SISR(zeta, plot=False):
    tau = np.zeros((2, m))  # vector of estimates
        
    def value(x,y,l):
        return y[l]-v+10*eta*np.log10(np.sqrt((x[0]-pos_vec[0,l])**2 + (x[3]-pos_vec[1,l])**2))

    # Initialization
    part = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]),size=N).T
    wgt_SISR = np.zeros((N,m))
    wgt_SISR[:,0] = np.prod(np.array([scipy.stats.norm.pdf(value(part,Y[:,0],l), 0, zeta) for l in range(6)]), axis=0)
    tau[0, 0] = np.sum(part[0, :]*wgt_SISR[:,0])
    tau[1 ,0] = np.sum(part[3, :]*wgt_SISR[:,0])

    # Propagation of the particles
    def indexx(x):
        return np.random.choice(a=range(len(P)), p=P[x,:])
    
    def ind_to_state(ind):
        return Z_state[ind]
    
    rng = np.random.default_rng()
    Z_index = rng.integers(0,5,size=N)
    Z = ind_to_state(Z_index).T

    # C_N_SISR = np.sum(wgt_SISR[:,0])/N
    for k in range(1,m):
        ind = np.random.choice(a=range(len(wgt_SISR)), size=N, replace=True, p=wgt_SISR[:,k-1]/np.sum(wgt_SISR[:,k-1]))
        part = part[:,ind]
        part = np.dot(Phi,part) + np.dot(Psi_Z,Z) + np.dot(Psi_W,np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2), size=N).T)
        wgt_SISR[:,k] = np.prod(np.array([scipy.stats.norm.pdf(value(part,Y[:,k],l), 0, zeta) for l in range(6)]),axis=0)
        print("Current iteration:", k, "weights", np.sum(wgt_SISR[:,k]))
        tau[0, k] = np.sum(part[0, :]*wgt_SISR[:,k])/np.sum(wgt_SISR[:,k])
        tau[1 ,k] = np.sum(part[3, :]*wgt_SISR[:,k])/np.sum(wgt_SISR[:,k])
        Z_index = np.array([indexx(Z_index[l]) for l in range(N)])
        Z = ind_to_state(Z_index).T
        # C_N_SISR = C_N_SISR*np.sum(wgt_SISR[:,k])/N
        
    Omega_list = np.sum(wgt_SISR, axis=0)
    C_N_SISR = (1/N**(m))*np.prod(Omega_list)


    if plot:
        plt.figure()
        plt.plot(tau[0, :], tau[1, :], '*')
        plt.plot(pos_vec[0, :], pos_vec[1, :], '*', color='black')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
    
    return np.log(C_N_SISR)/m
##################################################################

print(SISR(zeta=2))
# N_grid = 5
# zeta_grid = np.linspace(start=1.5, stop=3, num=N_grid)
# normalized_log_likelihood = np.zeros(N_grid)
# for i in range(N_grid):
#     zeta = zeta_grid[i]
#     print("Current zeta:", zeta)
#     likelihood = SISR(zeta)
#     normalized_log_likelihood[i] = np.log(likelihood)/m

# zeta_hat_ind = np.argmax(normalized_log_likelihood)
# zeta_hat = zeta_grid[zeta_hat_ind]
# print(normalized_log_likelihood)
# print("Zeta hat:", zeta_hat)
# print(SISR(zeta=zeta_hat, plot=True))