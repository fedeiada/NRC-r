import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt

# Speed of sound underwater
def c(T=20, s=35, z=20):
    return (1449.2 + 4.6 * T - 0.055 * (T ** 2) + 0.00029 * (T ** 3) + (1.34 - 0.01 * T) * (s - 35) + 0.16 * z)


# Defining Function (old one)
def f(m=12, N=512, M=2,pc=0.25, sea_cond=np.array([12, 35, 20, 300]), *, eta=1, B=4000, toh=1e-3, Rc=0.5):
    """Returns the cost function."""
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    return (np.log(m * (1 + pc) * N + B * (toh + td))
            - np.log(m) - np.log(Rc) - np.log(B) - np.log(N/eta)
            - np.log(np.log2(M)))

def g(m=12, N=512, M=2 , *, pc=0.25, sea_cond=np.array([12, 35, 40, 300]),toh=1e-3, B=4000):
    """Returns the gradient of the objective function.
    """
    # Helper variables
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh+td) * B
    p = 1 + pc
    dJdm = -1/m + ((N * p)/(L + N*m*p))
    dJdN = -1/N + ((m * p)/(L + N*m*p))
    dJdM = -1 / (M * np.log(M))
    return np.array([dJdm, dJdN, dJdM]).T

def h(m=12, N=512, M=2 , *,pc=0.25, sea_cond=np.array([12, 35, 20, 300]), toh=1e-3, B=4000):
    """Returns the Hessian of the objective function.
    """
    # Helper variables
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


# Implementing Newton Raphson Method

def newtonRaphson(m, N, M, e, iter):
    print('\n\n*** NEWTON RAPHSON METHOD IMPLEMENTATION ***')
    step = 1
    flag = 1
    condition = True
    x0 = np.array([m, N, M]).T
    x = np.zeros((3, iter+1))
    x[:, 0] = x0
    fx = [f(x0[0], x0[1], x0[2])]
    while condition:
        hesse = h(x0[0], x0[1], x0[2])
        # check if matrix is invertible: not singular
        if np.linalg.det(hesse) == 0:
            print('Impossible to invert!')
            flag = 0
            break

        x1 = x0.T - np.matmul(np.linalg.inv(h(x0[0], x0[1], x0[2])), g(x0[0], x0[1], x0[2]))
        print(f'Iteration-{step}, cost function value:{f(x1[0],x1[1],x1[2])} ')
        x0 = x1
        x[:, step] = x0
        step = step + 1
        fx.append(f(x0[0], x0[1], x0[2]))

        if step > iter:
            flag = 0
            break

        condition = abs(f(x0[0],x0[1],x0[2])) > e

    if flag == 1:
        print(f'\n Final cost is {fx[-1]}')
        fig, axs = plt.subplots(1, 2, figsize=(13, 8))
        axs[0].plot(range(0, step), x[0, :])
        axs[0].plot(range(0, step), x[1, :])
        axs[0].plot(range(0, step), x[2, :])
        axs[1].plot(range(0, step), fx)
        axs[0].grid(); axs[1].grid()
        plt.show()
    else:
        print(f'\nNot Convergent.')
        fig, axs = plt.subplots(1, 2, figsize=(13, 8))
        axs[0].plot(range(0, step), x[0, 0:step], label='m')
        axs[0].plot(range(0, step), x[1, 0:step], label='N')
        axs[0].plot(range(0, step), x[2, 0:step], label='M')
        axs[1].plot(range(0, step), fx)
        axs[0].legend(loc='upper right', ncol=1)
        axs[0].grid()
        axs[1].grid()
        plt.show()

Interactive = False
if Interactive:
    # Input Section
    x0 = input('Enter Guess: ')
    e = input('Tolerable Error:  ')
    iter = input('Maximum Step: ')
else:
    m = 5
    N = 128
    M = 8
    e = 0.00001
    iter =200

# Converting x0 and e to float
m = float(m)
N = float(N)
M = float(M)
e = float(e)

# Converting N to integer
iter = int(iter)


# Starting Newton Raphson Method
newtonRaphson(m, N, M, e, iter)

'''
OPTIMIZE using scipy function
'''
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