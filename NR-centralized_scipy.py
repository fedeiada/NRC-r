'''
OPTIMIZE using scipy function
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt

##### FLAG TO INCLUDE THE BARRIER TERM OR NOT #####
global barrier
barrier = False
###################################################
def b(pc, N, mu=0.2,  tau=0.025, B=4000):
    b = -(mu ** (-1)) * (np.log((1 - pc) * (pc - tau * (B / N))))
    return b

def c(T=20, s=35, z=20):
    return (1449.2 + 4.6 * T - 0.055 * (T ** 2) + 0.00029 * (T ** 3) + (1.34 - 0.01 * T) * (s - 35) + 0.16 * z)

def f(x, pc, sea_cond=np.array([12, 35, 20, 300]), eta=1, B=4000, toh=1e-3, Rc=0.5, mu = 0.2, tau = 0.025):
    """Returns the cost function."""
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    # add IPM (interior point method) for evaluate pc
    if barrier:
        b = (mu ** (-1)) * (np.log((1 - pc) * (pc - tau * (B / N))))
        return (-b + np.log(m * (1 + pc) * N + B * (toh + td))
                - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
                - np.log(np.log2(M)))
    else:
        return (np.log(m * (1 + pc) * N + B * (toh + td))
                - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
                - np.log(np.log2(M)))

def f_b(x, pc=0.25, sea_cond=np.array([12, 35, 20, 300]), *, eta=1, B=4000, toh=1e-3, Rc=0.5, mu = 0.2, tau = 0.025):
    """Returns the cost function."""
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    # add IPM (interior point method) for evaluate pc
    b = (mu ** (-1)) * (np.log((1 - pc) * (pc - tau * (B / N))))
    return (-b + np.log(m * (1 + pc) * N + B * (toh + td))
           - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
           - np.log(np.log2(M)))



def g(x, pc, sea_cond=np.array([12, 35, 40, 300]),toh=1e-3, B=4000, tau=0.025, mu=0.2):
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
        b = (B*tau)/((N**2)*mu*(pc-(B*tau)/N)) # barrier term
        dJdN = -1 / N + ((m * p) / (L + N * m * p)) + b
    else:
        dJdN = -1 / N + ((m * p) / (L + N * m * p))
    dJdm = -1/m + ((N * p)/(L + N*m*p))
    dJdM = -1 / (M * np.log(M))
    dJdp = -(N * m) / (L + N * m * p)
    return np.array([dJdm, dJdN, dJdM]).T

def h(x, *, pc=0.25, sea_cond=np.array([12, 35, 20, 300]), toh=1e-3, B=4000, mu=0.2, tau=0.025):
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
        b = ((B**2)*(tau**2))/((N**4)*mu*(pc - (B*tau)/N)**2) + (2*B*tau)/((N**3)*mu*(pc - (B*tau)/N)) # barrier term
        d2JdN2 = -((m ** 2) * (p ** 2)) / (L + N * m * p) ** 2 + (1 / (N ** 2)) - b
    else:
        d2JdN2 = -((m ** 2) * (p ** 2)) / (L + N * m * p) ** 2 + 1 / N ** 2
    d2Jdm2 = -((N**2)*(p**2))/(L + N*m*p)**2 + 1/m**2
    d2JdmdN = -((N*m)*(p**2))/(L + N*m*p)**2 + p/(L + N*m*p)
    d2Jdmdp = -((N**2)*m*p)/(L + N*m*p)**2 + N/(L + N*m*p)

    d2JdNdp = -(N*(m**2)*p)/(L + N*m*p)**2 + m/(L + N*m*p)
    d2JdM2 = 1/((M**2)*np.log(M)) + 1/((M)**2*np.log(M)**2)
    d2Jdp2 = -((N**2)*(m**2))/(L + N*m*p)**2
    return np.array([[d2Jdm2, d2JdmdN,  0],
                     [d2JdmdN, d2JdN2,  0],
                     [0   ,   0,  d2JdM2]])
def fun_update(xk):
    fun_evolution.append(f(xk, pc))




#opt.show_options()
x0 = [12, 512, 2]
Nx = 64  # Non-data subcarriers
B = 4000   # Bandwidth
k = 2  # rel doppler margin
v = 1  # doppler spread (hz)
tau = 0.025  # delay spread (s)
pc = 0.25  # cyclic prefix
bnds = ((1, None), (tau*B/pc, B/(k*v)), (2, None))
fun_evolution = []
res = opt.minimize(f, x0, args=(pc), callback=fun_update, bounds=bnds, jac=g, options={'maxiter':40, 'ftol': 10e-6, 'disp': False}) #fun_update(fun_evolution),
print(f"final value of OFDM parameters:\n m:{res.x[0]}\n N:{res.x[1]}\n M:{res.x[2]}\n")
print(f"success: {res.success}\n status:{res.message}\n")

#evolution of the cost function
plt.plot(range(res.nit), fun_evolution)
plt.grid()
plt.show()

'''
fun = []
m = []
N = []
M = []
pcrange = np.arange(0.1, 1.05, 0.05)
for p_c in pcrange:
    bnds = ((1, None), (tau * B / p_c, B / (k * v)), (2, None))
    res = opt.minimize(f, x0, args=(p_c), bounds=bnds, jac=g, options={'maxiter': 40, 'ftol': 10e-6, 'disp': False})
    fun.append(res.fun)
    m.append(res.x[0])
    N.append(res.x[1])
    M.append(res.x[2])

# plot comparing ranging of pc
fig, axs = plt.subplots(2,2, figsize=(13, 8))
fig.suptitle('ranging of value with different cyclic prefix')
axs[0,0].plot(pcrange, fun)
axs[0,0].set_title('cost function')
axs[0,1].plot(pcrange, m)
axs[0,1].set_title('m')
axs[1,0].plot(pcrange, N)
axs[1,0].set_title('N')
axs[1,1].plot(pcrange, M)
axs[1,1].set_title('M')
plt.savefig('multiplot.png')
plt.show()

# plot comparing evolution of cost/optimization variable
fig, axs = plt.subplots(3,1, figsize=(13, 8))
fig.suptitle('ranging of value with different cyclic prefix')
axs[0].plot(fun, m,)
axs[0].set_title('m')
axs[1].plot(fun, N)
axs[1].set_title('N')
axs[2].plot(fun, M)
axs[2].set_title('M')
#plt.savefig('multiplot.png')
plt.show()

#opt.show_options(solver='minimize', method='L-BFGS-B')

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

