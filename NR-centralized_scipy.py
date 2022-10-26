'''
OPTIMIZE using scipy function --> scipy.optimize.minimize
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
import math as mt
import itertools as it

##### FLAG TO INCLUDE THE BARRIER TERM OR NOT #####
global barrier
barrier = True
###################################################
def b(m, N, M, mu=0.2, eta=1):
    b = -(mu ** (-1)) * (np.log(-(m*(N/eta)*np.log2(M)*0.2*np.exp(-(3*100)/(2*(M-1)*0.5))-0.1)))
    return b

# function that calculate the speed of sound underwater
def c(T=20, s=35, z=20):
    return (1449.2 + 4.6 * T - 0.055 * (T ** 2) + 0.00029 * (T ** 3) + (1.34 - 0.01 * T) * (s - 35) + 0.16 * z)

# cost function
def f(x, pc, sea_cond=np.array([12, 35, 20, 300]), eta=1, B=4000, toh=1e-3, Rc=0.5, mu = 0.2, tau = 0.025):
    """Returns the cost function."""
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    # add IPM (interior point method) for evaluate pc
    if barrier:
        bterm = b(m, N, M)
        return (bterm + np.log(m * (1 + pc) * N + B * (toh + td))
                - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
                - np.log(np.log2(M)))
    else:
        return (np.log(m * (1 + pc) * N + B * (toh + td))
                - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
                - np.log(np.log2(M)))


# gradient of J0
def g(x, pc, sea_cond=np.array([12, 35, 40, 300]),toh=1e-3, B=4000, eta=0.2, mu=0.2):
    """Returns the gradient of the objective function.
    """
    # Helper variables
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh+td) * B
    p = 1 + pc
    if barrier:
        dJdN = -1 / N + ((m * p) / (L + N * m * p)) + \
               (m*np.exp(-300/(M - 1))*np.log(M))/(5*eta*mu*np.log(2)*((N*m*np.exp(-300/(M - 1))*np.log(M))/(5*eta*np.log(2)) - 1/10))
        dJdm = -1 / m + ((N * p) / (L + N * m * p)) + \
               (N * np.exp(-300 / (M - 1)) * np.log(M)) / (5 * eta * mu * np.log(2) * ((N * m * np.exp(-300 / (M - 1)) * np.log(M)) / (5 * eta * np.log(0.2)) - 1 / 10))
        dJdM = -1 / (M * np.log(M)) + \
               ((N * m * np.exp(-300 / (M - 1))) / (5 * M * eta * np.log(2)) + (60 * N * m * np.exp(-300 / (M - 1)) * np.log(M)) / (eta * np.log(2) * ((M - 1) ** 2))) / (
                           mu * ((N * m * np.exp(-300 / (M - 1)) * np.log(M)) / (5 * eta * np.log(2)) - 1 / 10))

    else:
        dJdN = -1 / N + ((m * p) / (L + N * m * p))
        dJdm = -1/m + ((N * p)/(L + N*m*p))
        dJdM = -1 / (M * np.log(M))
    return np.array([dJdm, dJdN, dJdM]).T

# hessian of J0
def h(x, pc,*, sea_cond=np.array([12, 35, 20, 300]), toh=1e-3, B=4000, mu=0.2, tau=0.025, eta=1, Rc=0.5, pb=0.1):
    """Returns the Hessian of the objective function.
    """
    # Helper variables
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh + td) * B
    p = 1 + pc
    if barrier:
        d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (m ** 2) + \
                 (400*(N**2)*np.exp((20*N*m*np.exp(-300/(Rc*2*(M - 1)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-600/(Rc*2*(M - 1)))*(np.log(M)**2))/((eta**2)*(np.log(2)**2))
        d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p) + \
                  (20 * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) +\
                  (400*N*m*np.exp((20*N*m*np.exp(-300/(Rc*2*(M - 1)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-600/(Rc*2*(M - 1)))*(np.log(M)**2))/((eta**2)*(np.log(2)**2))
        d2JdmdM = (20*N*np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-300/(Rc*(2*M - 2))))/(M*eta*np.log(2)) + \
                  (20*N*np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-300/(Rc*(2*M - 2)))*np.log(M)*((20*N*m*np.exp(-300/(Rc*(2*M - 2))))/(M*eta*np.log(2)) +
                  (12000*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc*eta*np.log(2)*(2*M - 2)**2)))/(eta*np.log(2)) + (12000*N*np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc*eta*np.log(2)*(2*M - 2)**2)
        d2JdN2 = -((m ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + (1 / (N ** 2)) + \
                 (400 * (m**2) * np.exp((20 * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (np.log(M) ** 2)) / ((eta ** 2) * (np.log(2) ** 2))
        d2JdNdM = (20*m*np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-300/(Rc*(2*M - 2))))/(M*eta*np.log(2)) + \
                  (20*N*np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-300/(Rc*(2*M - 2)))*np.log(M)*((20*N*m*np.exp(-300/(Rc*(2*M - 2))))/(M*eta*np.log(2)) +
                  (12000*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc*eta*np.log(2)*(2*M - 2)**2)))/(eta*np.log(2)) + (12000*N*np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc*eta*np.log(2)*(2*M - 2)**2)
        d2JdM2 = 1 / ((M ** 2) * np.log(M)) + 1 / ((M) ** 2 * np.log(M) ** 2) + \
                 np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*((20*N*m*np.exp(-300/(Rc*(2*M - 2))))/(M*eta*np.log(2)) + (12000*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc*eta*np.log(2)*(2*M - 2)**2))**2 \
                 - np.exp((20*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(eta*np.log(2)) - 100*pb)*((20*N*m*np.exp(-300/(Rc*(2*M - 2))))/(M**2*eta*np.log(2)) - (24000*N*m*np.exp(-300/(Rc*(2*M - 2))))/(M*Rc*eta*np.log(2)*(2*M - 2)**2) + (48000*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc*eta*np.log(2)*(2*M - 2)**3) - (7200000*N*m*np.exp(-300/(Rc*(2*M - 2)))*np.log(M))/(Rc**2*eta*np.log(2)*(2*M - 2)**4))
        return np.array([[d2Jdm2, d2JdmdN, d2JdmdM],
                         [d2JdmdN, d2JdN2, d2JdNdM],
                         [d2JdmdM, d2JdNdM, d2JdM2]])

    else:
        d2JdN2 = -((m ** 2) * (p ** 2)) / (L + N * m * p) ** 2 + 1 / N ** 2
        d2Jdm2 = -((N**2)*(p**2))/(L + N*m*p)**2 + 1/m**2
        d2JdmdN = -((N*m)*(p**2))/(L + N*m*p)**2 + p/(L + N*m*p)
        d2JdM2 = 1/((M**2)*np.log(M)) + 1/((M)**2*np.log(M)**2)

        return np.array([[d2Jdm2, d2JdmdN, 0],
                        [d2JdmdN, d2JdN2,  0],
                        [  0   ,   0,  d2JdM2]])

# callback function: save the actual cost at the k-th iteration
def fun_update(xk, opt):
    fun_evolution.append(f(xk, pc))

def BER(x):
    return np.log(x[0]) + np.log(2**x[1]) + np.log(np.log2(x[2])) + 2 * (np.log(0.2) - ((3 * 100) / (2 * (x[2] - 1))))


#opt.show_options()
x0 = [12, 512, 2]
Nx = 64  # Non-data subcarriers
B = 4000   # Bandwidth
k = 2  # rel doppler margin (3) not higher than 5 or under 1
v = 1  # doppler spread (hz)
tau = 0.025  # delay spread (s)
pc = 0.25  # cyclic prefix

if barrier:
    bnds = ((1, 30), (tau*B/pc, B/(k*v)), (2, 64))
else:
    bnds = ((1, 30), (tau*B/pc, B/(k*v)), (2, 64))
fun_evolution = []
cons = ({'type': 'ineq', 'fun': lambda x: -(np.log(x[0]) + np.log(x[1]) + np.log(np.log2(x[2]))+2*(np.log(0.2)-((3*100)/(2*(x[2]-1))))-np.log(0.2))})       # prob_loss = 0.1, eta = 1, Rc=0.5
res = opt.minimize(f,
                   x0,
                   args=(pc),
                   method='trust-constr',
                   #constraints=cons,
                   bounds=bnds,
                   callback=fun_update,
                   jac=g,
                   hess=h,
                   tol=1e-3,
                   options={'maxiter':2000, 'disp': True})


'''
Quantization: need quantized value. round the result and evaluate if it's better take the upper or lower value
'''

m_up = mt.ceil(res.x[0])
m_low = int(res.x[0])
M_up = mt.ceil(res.x[2])
M_low = int(res.x[2])
N_low = int(np.log2(res.x[1]))

queue = list(it.permutations((m_low,m_up,M_low,M_up,N_low),3))

xx = [m_low, N_low, M_up]
ber = BER(xx)
cost_function = f([m_low, N_low, M_low], 0.25)
if BER([m_up, N_low, M_up]) <= ber:
    ber = BER(m_up, N_low, M_up)
    xx[0] = m_up
if BER([xx[0], N_low, M_low]) <= ber:
    ber = BER([xx[0], N_low, M_low])
    xx[2] = M_low
    cost_function = f([xx[0], N_low, xx[2]], 0.25)
if BER([xx[0], 10, xx[2]]) <= ber:
    print('GG')


print(f"final value of OFDM parameters:\n m:{res.x[0]}\n N:{res.x[1]}\n M:{res.x[2]}\n")
print(f"success: {res.success}\n status:{res.message}\n")

#evolution of the cost function
plt.plot(range(res.nit), fun_evolution)
plt.xlabel('Number of iterations')
plt.ylabel('cost function')
plt.grid()
plt.show()
plt.savefig('optim.png')

# make some plot ranging different possible values of the cyclic prefix 
fun = []
m = []
N = []
M = []
pcrange = np.linspace(0.1, 1, 50)
for p_c in pcrange:
    bnds = ((1, None), (tau * B/p_c, B/(k * v)), (2, None))
    cons = ({'type': 'ineq', 'fun': lambda x: -(np.log(x[0]) + np.log(x[1]) + np.log(np.log2(x[2])) + 2 * (np.log(0.2) - ((3 * 100) / (2 * (x[2] - 1)))) - np.log(0.1))})  # prob_loss = 0.1, eta = 1, Rc=0.5
    #res2 = opt.minimize(f, x0, args=(pc), method='SLSQP', callback=fun_update, constraints=cons, bounds=bnds, jac=g, options={'maxiter': 40, 'ftol': 10e-8, 'disp': True})

    res2 = opt.minimize(f,
                        x0,
                        args=(p_c),
                        method='trust-constr',
                        constraints=cons,
                        bounds=bnds,
                        jac=g,
                        hess=h,
                        tol=1e-3,
                        options={'maxiter': 1500, 'disp': False})
    fun.append(res2.fun)
    m.append(res2.x[0])
    N.append(res2.x[1])
    M.append(res2.x[2])

# plot comparing ranging of pc
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('ranging of value with different cyclic prefix')
axs[0,0].plot(pcrange, fun)
axs[0,0].set_xlabel('pc')
axs[0,0].set_ylabel('cost function')
axs[0,1].plot(pcrange, m)
axs[0,1].set_xlabel('pc')
axs[0,1].set_ylabel('m')
axs[1,0].plot(pcrange, N)
axs[1,0].set_xlabel('pc')
axs[1,0].set_ylabel('N')
axs[1,1].plot(pcrange, M)
axs[1,1].set_xlabel('pc')
axs[1,1].set_ylabel('M')
plt.savefig('ofdm_pc.png')
plt.show()

# plot comparing evolution of cost/optimization variable
fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
fig.suptitle('ranging of OFDM value with different cyclic prefix over different J0')
axs[0].plot(fun, m,)
axs[0].set_ylabel('m')
axs[1].plot(fun, N)
axs[1].set_ylabel('N')
axs[2].plot(fun, M)
axs[2].set_ylabel('M')
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.xlabel('cost function')
plt.savefig('ofdm_costf.png')
plt.show()

#'''
#opt.show_options(solver='minimize', method='L-BFGS-B')

'''
# unbounded optimization with Newton-CG
res = opt.minimize(f, x0, method='Newton-CG', jac=g, hess=h ,options={'maxiter':40, 'xtol': 10e-5, 'disp': True})
print(f"final value of OFDM parameters:\n m:{res.x[0]}\n N:{res.x[1]}\n M:{res.x[2]}\n")
print(f"success: {res.success}\n status:{res.message}")


l_b = 0.025 * (4000/res.x[1])
Nrange= range(128, 646)
pc_range = np.arange(l_b+0.05, 1, 0.04)
cost = []
cost2 = []
for pc in pc_range:
    cost.append(f_b(x=res.x, pc=pc))
plt.plot(pc_range, cost)
plt.show()

for N in Nrange:
    x_pass = [res.x[0], N, res.x[2]]
    cost2.append(f_b(x=x_pass))
plt.plot(Nrange, cost2)
plt.show()
'''

