'''
OPTIMIZE using scipy function
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt


def c(T=20, s=35, z=20):
    return (1449.2 + 4.6 * T - 0.055 * (T ** 2) + 0.00029 * (T ** 3) + (1.34 - 0.01 * T) * (s - 35) + 0.16 * z)

def f(x,pc=0.25, sea_cond=np.array([12, 35, 20, 300]), *, eta=1, B=4000, toh=1e-3, Rc=0.5):
    """Returns the cost function."""
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    return (np.log(m * (1 + pc) * N + B * (toh + td))
            - np.log(m) - np.log(Rc) - np.log(B) - np.log(N/eta)
            - np.log(np.log2(M)))

def g(x, *, pc=0.25, sea_cond=np.array([12, 35, 40, 300]),toh=1e-3, B=4000):
    """Returns the gradient of the objective function.
    """
    # Helper variables
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh+td) * B
    p = 1 + pc
    dJdm = -1/m + ((N * p)/(L + N*m*p))
    dJdN = -1/N + ((m * p)/(L + N*m*p))
    dJdM = -1 / (M * np.log(M))
    return np.array([dJdm, dJdN, dJdM]).T

def h(x, *, pc=0.25, sea_cond=np.array([12, 35, 20, 300]), toh=1e-3, B=4000):
    """Returns the Hessian of the objective function.
    """
    # Helper variables
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh + td) * B
    p = 1 + pc
    d2Jdm2 = -((N**2)*(p**2))/(L + N*m*p)**2 + 1/m**2
    d2JdmdN = -((N*m)*(p**2))/(L + N*m*p)**2 + p/(L + N*m*p)
    d2Jdmdp = -((N**2)*m*p)/(L + N*m*p)**2 + N/(L + N*m*p)
    d2JdN2 = -((m**2)*(p**2))/(L + N*m*p)**2 + 1/N**2
    d2JdNdp = -(N*(m**2)*p)/(L + N*m*p)**2 + m/(L + N*m*p)
    d2JdM2 =  1/((M**2)*np.log(M)) + 1/((M)**2*np.log(M)**2)
    d2Jdp2 = -((N**2)*(m**2))/(L + N*m*p)**2
    return np.array([[d2Jdm2, d2JdmdN,  0],
                     [d2JdmdN, d2JdN2,  0],
                     [0     ,   0,  d2JdM2]])

x0 = [12, 512, 2]
res = opt.minimize(f, x0, method='Newton-CG', jac=g, hess=h, options={'xtol': 1e-8, 'disp': True})