import numpy as np
import matplotlib.pyplot as plt

# optimal point found through matlab : 12.4471, 538.8997, 1, 0.3386

##### function definition #####
# change speed of sound depending on sea condition
def c(T=20, s=35, z=20):
    return (1449.2 + 4.6 * T - 0.055 * (T ** 2) + 0.00029 * (T ** 3) + (1.34 - 0.01 * T) * (s - 35) + 0.16 * z)


# cost function
def cost_1(m=12.4471, N=800, pc=0.3386, M=1.2, sea_cond=np.array([12, 35, 20, 300]), *, eta=1, B=4000, toh=1e-3, Rc=0.5, mu=0.2, tau=0.025):
    """Returns the cost function."""
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    b = (mu ** (-1)) * (np.log((1 - pc) * (pc - tau * (B / N))))
    return (-b + np.log(m * (1 + pc) * N + B * (toh + td))
            - np.log(m) - np.log(Rc) - np.log(B) - np.log(N/eta)
            - np.log(np.log2(M)))


def cost_2(m=3, N=512, pc=0.25, M=4, sea_cond=np.array([12, 35, 20, 300]), *, eta=1, B=4000, toh=1e-3, Rc=0.5):
    """Returns the cost function."""
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    u = [1, 1, 1]  # weights
    tau = 9
    delta_f = B / N
    ni = 10
    k = 3
    Pn = 40
    return (np.log(m * (1 + pc) * N + B * (toh + td))
            - np.log(m) - np.log(Rc) - np.log(B) - np.log(N /eta)
            - np.log(np.log2(M)) + np.log(1 + (u[0] * tau * delta_f + u[1] ** ni * k + u[2] * Pn)))


################################################

# set value for the sea condition

temp = 15  # celsius
salinity = 35  # parts per thousand
depth = 50  # meters
r = 100  # meters

params = np.array([temp, salinity, depth, r])
'''
mrange = np.arange(1, 41)
Nrange = 2 ** np.arange(3, 15)
pcrange = np.linspace(0, 1)
Mrange = 2 ** np.arange(1, 10)
'''
mrange = np.arange(1, 41)
Nrange = 2 ** np.arange(3, 16)
pcrange = np.linspace(0, 1)
Mrange = 2 ** np.arange(1, 13)
plt.figure(figsize=[11.2, 8.4])
# Collect all variables in lists
xlabs = ['Packet count $m$', 'Subcarrier count $N$',
         'Cyclic prefix fraction $p_c$', 'Constellation size $M$',
         'TX power $p_{tx}$ (dB re. 1 µPa @ 1 m)']
rngs = [mrange, Nrange, pcrange, Mrange]
kws = [{'m': mrange}, {'N': Nrange}, {'pc': pcrange}, {'M': Mrange}]

for it in range(4):
    plt.subplot(2, 2, it + 1)
    plt.plot(rngs[it], cost_1(**kws[it], sea_cond=params))
    plt.xlabel(xlabs[it])
    plt.xscale('log' if it % 2 == 1 else 'linear')
    plt.grid()
    plt.ylabel('Cost function')
    plt.savefig('range of cost function.png')
plt.show()


################## look at eigenvalues in the optimal point ##############################
def gradient(m, N, M ,pc, *, sea_cond=np.array([10, 35, 40, 300]),toh=1e-3, B=4000, tau=0.025, mu=0.2 ):
    """Returns the gradient of the objective function.
    """
    # Helper variables
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    print(td)
    L = (toh+td) * B
    p = 1 + pc
    dJdm = -1/m +((N * p)/(L + N*m*p))
    dJdN = -1/N +((m * p)/(L + N*m*p)) +(B*tau)/((N**2)*mu*(pc-(B*tau)/N))
    dJdM = -1 / (M * np.log(M))
    dJdp = -(N*m) / (L + N*m*p)
    return np.array([dJdm, dJdN, dJdM]).T

def hessian(m, N, M ,pc, *, sea_cond=np.array([12, 35, 20, 300]), toh=1e-3, B=4000, tau=0.025, mu=0.2):
    """Returns the Hessian of the objective function.
    """
    # Helper variables
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh + td) * B
    p = 1 + pc
    d2Jdm2 = -((N**2)*(p**2))/(L + N*m*p)**2 + 1/m**2
    d2JdmdN = -((N*m)*(p**2))/(L + N*m*p)**2 + p/(L + N*m*p)
    d2Jdmdp = -((N**2)*m*p)/(L + N*m*p)**2 + N/(L + N*m*p)
    d2JdN2 = -((m**2)*(p**2))/(L + N*m*p)**2 + 1/N**2 -(B**2*tau**2)/((N**4)*mu*(pc - (B*tau)/N)**2) + (2*B*tau)/((N**3)*mu*(pc - (B*tau)/N))
    d2JdNdp = -(N*(m**2)*p)/(L + N*m*p)**2 + m/(L + N*m*p)
    d2JdM2 =  1/((M**2)*np.log(M)) + 1/((M)**2*np.log(M)**2)
    d2Jdp2 = -((N**2)*(m**2))/(L + N*m*p)**2
    return np.array([[d2Jdm2, d2JdmdN,  0],
                     [d2JdmdN, d2JdN2,  0],
                     [0     ,   0,  d2JdM2]])
    '''
    return np.array([[d2Jdm2, d2JdmdN,  0, d2Jdmdp],
                     [d2JdmdN, d2JdN2,  0, d2JdNdp],
                     [0      ,   0,  d2JdM2,     0],
                     [d2Jdmdp, d2JdNdp, 0, d2Jdp2]])
    '''

# optimal point
m = 22.4262  #25
N = 1024.8211 #220.15
M = 4 #26.2
pc = 0.3369 #0.2
grad = gradient(m, N, M, pc)
hesse = hessian(m, N, M, pc)
eigval, eigvec = np.linalg.eigh(hesse)
print(f"Hessian matrix \n{hesse}")
print(f"Eigenvalues are\n{eigval}\n and map to the columns of\n{eigvec}")

'''
eig2val, eig2vec = np.linalg.eigh(hesse[:-1, :-1])
print(f"With cyclic prefix fraction fixed, eigenvalues are\n{eig2val}\n"
      f"and map to the columns of\n{eig2vec}")
'''

## Gershgorin theorem to look at convexity
def gershgorin_disks(A):
    """Finds the Gershgorin disks of a square matrix."""
    if A.ndim < 2:
        raise TypeError("A must have at least two axes")
    elif A.shape[-1] != A.shape[-2]:
        raise TypeError("Expected A to be square along the last two"
                        f" axes, but it has shape {A.shape[-2:]}")
    # Centra are given by the diagonal elements of A
    centra = np.diagonal(A, axis1=-2, axis2=-1)
    # Radii are given by the row-wise sum of the magnitude of the off-diagonals
    radii = np.sum(np.abs(A), axis=-1) - np.abs(centra)
    return (centra, radii)

def gen_circle(centrum, radius, npoints=101):
    """Construct a circle."""
    angles = np.linspace(-np.pi, np.pi, npoints)
    try:
        angles = np.tile(angles[:, None], [1, centrum.shape[0]])
    except IndexError:
        pass
    except AttributeError:
        pass
    return (np.cos(angles) * radius + centrum.real,
            np.sin(angles) * radius + centrum.imag)


centra, radii = gershgorin_disks(hesse)
plt.figure(figsize=[7.8, 2.4])
for it in range(len(centra)):
    plt.subplot(1, 3, 1 + it)
    plt.plot(*gen_circle(centra[it], radii[it]))
    if it == 0:
        plt.plot(*gen_circle(centra[1], radii[1]))
    plt.axvline(x=0, color="tab:gray")
    plt.xlabel("Real part")
    plt.ylabel("Imag. part")
plt.show()

'''
centra2, radii2 = gershgorin_disks(hesse[0:3, 0:3])
tr2 = np.trace(hesse[0:3, 0:3])
prod2 = np.linalg.det(hesse[0:3, 0:3])
print(f"x^2{-tr2:+.3e}x{prod2:+.3e} = 0 is the characteristic equation with pc fixed")

plt.figure(figsize=[5.2, 2.4])
for it in range(len(centra2)):
    plt.subplot(1, 3, 1 + it)
    plt.plot(*gen_circle(centra2[it], radii2[it]))
    if it == 0:
        plt.plot(*gen_circle(centra2[1], radii2[1]))
    plt.axvline(x=0, color="tab:gray")
    plt.xlabel("Real part")
    plt.ylabel("Imag. part")
plt.show()
'''
