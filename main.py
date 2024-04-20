import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import random

random.seed(23)

############ PROBLEM 1 ############

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
    W = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2))
    X_sim[:,n+1] = Phi @ X_sim[:,n] +  Psi_Z @ Z_sim[:,n] + Psi_W @ W
  
# Plot the trajectory  
plt.figure()
plt.plot(X_sim[0, :], X_sim[3, :])
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()

############ PROBLEM 3 ############

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

# Data processing
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

def value(x,y,l):
    return y[l]-v+10*eta*np.log10(np.sqrt((x[0]-pos_vec[0,l])**2 + (x[3]-pos_vec[1,l])**2))

def exp_and_normalize(log_w):
    # with implementation to avoid underflow after the exponential
    L = np.max(log_w)
    w = np.exp(log_w - L)
    return w/np.sum(w)

# Initialization of the particles, weights and estimates
part = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]),size=N).T
log_wgt_SIS = np.zeros((N,m))
normalized_wgt_SIS = np.zeros((N,m))
log_wgt_SIS[:,0] = np.sum(np.array([scipy.stats.norm.logpdf(value(part,Y[:,0],l), 0, zeta) for l in range(6)]), axis=0)
normalized_wgt_SIS[:,0] = exp_and_normalize(log_wgt_SIS[:,0])
tau[0, 0] = np.sum(part[0, :]*normalized_wgt_SIS[:,0])
tau[1 ,0] = np.sum(part[3, :]*normalized_wgt_SIS[:,0])

# functions for the markov chain Z
def indexx(x):
    return np.random.choice(a=range(len(P)), p=P[x,:])

def ind_to_state(ind):
    return Z_state[ind]

#Initialization of the markov chain Z
rng = np.random.default_rng()
Z_index = rng.integers(0,5,size=N)
Z = ind_to_state(Z_index).T

# Propagation 
for k in range(1,m):
    part = np.dot(Phi,part) + np.dot(Psi_Z,Z) + np.dot(Psi_W,np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2), size=N).T)
    log_wgt_SIS[:,k] = np.sum(np.array([scipy.stats.norm.logpdf(value(part,Y[:,k],l), 0, zeta) for l in range(6)]),axis=0) + log_wgt_SIS[:, k-1]
    normalized_wgt_SIS[:,k] = exp_and_normalize(log_wgt_SIS[:,k])
    tau[0, k] = np.sum(part[0, :]*normalized_wgt_SIS[:,k])
    tau[1 ,k] = np.sum(part[3, :]*normalized_wgt_SIS[:,k])
    Z_index = np.array([indexx(Z_index[l]) for l in range(N)])
    Z = ind_to_state(Z_index).T

# Compute the ESS at different times
m_vector = [10, 50, 100, 200, 500]
CV_square = np.array([N*np.sum((normalized_wgt_SIS[:,m]-1/N)**2) for m in m_vector])
ESS = N/(1+CV_square)
print(ESS)

# Plot the trajectory of the estimates
plt.figure()
plt.plot(tau[0, :], tau[1, :], '*')
plt.plot(pos_vec[0, :], pos_vec[1, :], '*', color='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Plot the importance-weight distribution
plt.figure()
bin_pos = np.linspace(-400,0,20)
H = bin_pos
plt.hist([log_wgt_SIS[:,m] for m in m_vector], 
         label=['m = '+str(m) for m in m_vector], bins=30) 
plt.legend()
plt.xlabel('Importance weights (natural logarithm)')
plt.ylabel('Absolute frequency')
plt.title('Importance-weight distribution SIS')
plt.show()


############ PROBLEM 4 ############

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

# Data processing
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
    
def value(x,y,l):
    return y[l]-v+10*eta*np.log10(np.sqrt((x[0]-pos_vec[0,l])**2 + (x[3]-pos_vec[1,l])**2))

# Initialization of the particles, weights and estimates
part = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]),size=N).T
wgt_SISR = np.zeros((N,m))
wgt_SISR[:,0] = np.prod(np.array([scipy.stats.norm.pdf(value(part,Y[:,0],l), 0, zeta) for l in range(6)]), axis=0)
tau[0, 0] = np.sum(part[0, :]*wgt_SISR[:,0])
tau[1 ,0] = np.sum(part[3, :]*wgt_SISR[:,0])


# functions for the Markov chain Z

def indexx(x):
    return np.random.choice(a=range(len(P)), p=P[x,:])

def ind_to_state(ind):
    return Z_state[ind]

# Initialization of the Markov chain Z
rng = np.random.default_rng()
Z_index = rng.integers(0,5,size=N)
Z = ind_to_state(Z_index).T

# Initialization of the most probable state
Z_maxoccur = np.zeros(m)
Z_maxoccur[0] = np.argmax(np.bincount(Z_index, minlength=5))

# Propagation
for k in range(1,m):
    ind = np.random.choice(a=range(len(wgt_SISR)), size=N, replace=True, p=wgt_SISR[:,k-1]/np.sum(wgt_SISR[:,k-1]))
    part = part[:,ind]
    part = np.dot(Phi,part) + np.dot(Psi_Z,Z) + np.dot(Psi_W,np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2), size=N).T) 
    wgt_SISR[:,k] = np.prod(np.array([scipy.stats.norm.pdf(value(part,Y[:,k],l), 0, zeta) for l in range(6)]),axis=0)
    tau[0, k] = np.sum(part[0, :]*wgt_SISR[:,k])/np.sum(wgt_SISR[:,k])
    tau[1 ,k] = np.sum(part[3, :]*wgt_SISR[:,k])/np.sum(wgt_SISR[:,k])
    Z_index = np.array([indexx(Z_index[l]) for l in range(N)])
    Z = ind_to_state(Z_index).T
    Z_maxoccur[k] = np.argmax(np.bincount(Z_index, minlength=5))
    
# Plot the trajectory of the estimates
plt.figure()
plt.plot(tau[0, :], tau[1, :], '*')
plt.plot(pos_vec[0, :], pos_vec[1, :], '*', color='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Plot the importance-weight distribution
plt.figure()
m_vector = [10, 50, 100, 200, 500]
bin_pos = np.linspace(-400,0,20)
H = bin_pos
plt.hist([np.log(wgt_SISR[:,m]) for m in m_vector], 
         label=['m = '+str(m) for m in m_vector], bins=30) 
plt.legend()
plt.xlabel('Importance weights (natural logarithm)')
plt.ylabel('Absolute frequency')
plt.title('Importance-weight distribution SISR')
plt.show()

# Plot the most probable state over time
plt.figure()
plt.plot(np.arange(0,m),Z_maxoccur,'o')
plt.xlabel('Time')
plt.ylabel('Most Probable State')
plt.title('Most Probable State over Time')
plt.xticks(np.arange(0, m, step=50))
plt.yticks(np.arange(0, 5))
plt.grid(True)
plt.show()


############ PROBLEM 5 ############

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


# Data processing
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

# Sequential importance sampling for a zeta value
def SISR(zeta, plot=False):
    tau = np.zeros((2, m))  # vector of estimates
        
    def value(x,y,l):
        return y[l]-v+10*eta*np.log10(np.sqrt((x[0]-pos_vec[0,l])**2 + (x[3]-pos_vec[1,l])**2))

    # Initialization of the particles, weights and estimates
    part = np.random.multivariate_normal(mean=np.zeros(6), cov=np.diag([500, 5, 5, 200, 5, 5]),size=N).T
    wgt_SISR = np.zeros((N,m))
    wgt_SISR[:,0] = np.prod(np.array([scipy.stats.norm.pdf(value(part,Y[:,0],l), 0, zeta) for l in range(6)]), axis=0)
    tau[0, 0] = np.sum(part[0, :]*wgt_SISR[:,0])
    tau[1 ,0] = np.sum(part[3, :]*wgt_SISR[:,0])

    # functions for the Markov chain Z
    def indexx(x):
        return np.random.choice(a=range(len(P)), p=P[x,:])
    
    def ind_to_state(ind):
        return Z_state[ind]
    
    # Initialization of the markov chain Z
    rng = np.random.default_rng()
    Z_index = rng.integers(0,5,size=N)
    Z = ind_to_state(Z_index).T

    # Propagation
    for k in range(1,m):
        ind = np.random.choice(a=range(len(wgt_SISR)), size=N, replace=True, p=wgt_SISR[:,k-1]/np.sum(wgt_SISR[:,k-1]))
        part = part[:,ind]
        part = np.dot(Phi,part) + np.dot(Psi_Z,Z) + np.dot(Psi_W,np.random.multivariate_normal(mean=np.zeros(2), cov=sigma**2*np.eye(2), size=N).T)
        wgt_SISR[:,k] = np.prod(np.array([scipy.stats.norm.pdf(value(part,Y[:,k],l), 0, zeta) for l in range(6)]),axis=0)
        tau[0, k] = np.sum(part[0, :]*wgt_SISR[:,k])/np.sum(wgt_SISR[:,k])
        tau[1 ,k] = np.sum(part[3, :]*wgt_SISR[:,k])/np.sum(wgt_SISR[:,k])
        Z_index = np.array([indexx(Z_index[l]) for l in range(N)])
        Z = ind_to_state(Z_index).T
    
    # Compute the estimate of the likelihood 
    Omega_list = np.sum(wgt_SISR, axis=0)
    C_N_SISR = (1/N**(m))*np.prod(Omega_list)
    
    # Plot the trajectory of the estimates
    if plot:
        plt.figure()
        plt.plot(tau[0, :], tau[1, :], '*')
        plt.plot(pos_vec[0, :], pos_vec[1, :], '*', color='black')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
    
    # Return the estimate of the normalized log likelihood
    return np.log(C_N_SISR)/m