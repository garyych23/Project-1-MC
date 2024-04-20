import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import random

random.seed(23)

# Load data
pos_vec_file = open('stations.txt', "r")
Y_file = open('RSSI-measurements.txt', "r")

# Parameters
m = 501
dt = 0.5
alpha = 0.6
sigma = 0.5
N = 10000
v = 90
eta = 3
zeta = 1.5

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

# Sequential importance sampling
tau = np.zeros((2, m))  # vector of estimates

def findDist(l, x, pos_vec):
    p = np.array([x[0], x[3]]) - np.array([pos_vec[0,l-1], pos_vec[1,l-1]])
    dist = np.sqrt(p[0]**2 + p[1]**2)
    return dist

def p(x, y):
    values = np.array([y[k]-v+10*eta*np.log10(findDist(k+1,x,pos_vec)) for k in range(len(y))])
    res = np.array([scipy.stats.norm.pdf(values[k], 0, zeta) for k in range(len(y))])
    return np.prod(res)

def logp(x,y):
    values = np.array([y[k]-v+10*eta*np.log10(findDist(k+1,x,pos_vec)) for k in range(len(y))])
    res = np.array([scipy.stats.norm.logpdf(values[k], 0, zeta) for k in range(len(y))])
    return np.sum(res)
    
def value(x,y,l):
    return y[l]-v+10*eta*np.log10(np.sqrt((x[0]-pos_vec[0,l])**2 + (x[3]-pos_vec[1,l])**2))

def exp_and_normalize(log_w):
    # with implementation to avoid underflow after the exponential
    L = np.max(log_w)
    w = np.exp(log_w - L)
    return w/np.sum(w)

# Initialization
part = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]),size=N).T
print(part.shape)
log_wgt_SIS = np.zeros((N,m))
normalized_wgt_SIS = np.zeros((N,m))
log_wgt_SIS[:,0] = np.sum(np.array([scipy.stats.norm.logpdf(value(part,Y[:,0],l), 0, zeta) for l in range(6)]))
normalized_wgt_SIS[:,0] = exp_and_normalize(log_wgt_SIS[:,0])
tau[0, 0] = np.sum(part[0, :]*normalized_wgt_SIS[:,0])
tau[1 ,0] = np.sum(part[3, :]*normalized_wgt_SIS[:,0])


# Propagation of the particles
def ind_to_state(ind):
    return Z_state[ind]

def indexx(x):
    return np.random.choice(a=range(len(P)), p=P[x,:])
    

rng = np.random.default_rng()
Z_index = rng.integers(0,5,size=N)
Z = ind_to_state(Z_index).T
for k in range(1,m):
    ind = random.choices(range(len(normalized_wgt_SIS)), weights=normalized_wgt_SIS, k=N)
    part = [part[:,i] for i in ind]
    #part = part[:,ind]
    part = np.dot(Phi,part) + np.dot(Psi_Z,Z) + np.dot(Psi_W,np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2), size=N).T)
    log_wgt_SIS[:,k] = np.sum(np.array([scipy.stats.norm.logpdf(value(part,Y[:,k],l), 0, zeta) for l in range(6)])) + log_wgt_SIS[:, k-1]
    normalized_wgt_SIS[:,k] = exp_and_normalize(log_wgt_SIS[:,k])
    tau[0, k] = np.sum(part[0, :]*normalized_wgt_SIS[:,k])
    tau[1 ,k] = np.sum(part[3, :]*normalized_wgt_SIS[:,k])
    Z_index = np.array([indexx(Z_index[l]) for l in range(N)])
    Z = ind_to_state(Z_index).T
    print(np.max(np.abs(normalized_wgt_SIS)))

plt.figure()
plt.plot(tau[0, :], tau[1, :], '*')
plt.plot(pos_vec[0, :], pos_vec[1, :], '*', color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plt.figure()
bin_pos = np.linspace(-400,0,20)
H = bin_pos
plt.hist(log_wgt_SIS[:, -1], bins=30)
plt.xlabel('Importance Weights')
plt.ylabel('Frequency')
plt.title('Histogram of Importance Weights')
plt.show()

