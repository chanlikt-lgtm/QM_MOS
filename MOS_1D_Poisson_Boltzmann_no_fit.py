
# 1-D MOS Poisson-Boltzmann solver
# No fitting parameters.
# Surface potential is obtained from VG through the oxide boundary condition.

import numpy as np
import matplotlib.pyplot as plt

q = 1.602176634e-19
eps0 = 8.8541878128e-14
eps_si = 11.7*eps0
eps_ox = 3.9*eps0
VT = 0.02585

NA = 1e16          # cm^-3
ni = 1e10
tox = 10e-7        # cm
VG = 1.0
VFB = 0.0

Cox = eps_ox/tox

# Silicon depth
Xmax = 1e-4
N = 400
dx = Xmax/(N-1)

x = np.linspace(0,Xmax,N)

# Initial potential
psi = np.ones(N)*0.3
psi[-1] = 0.0

def charge(p):
    # p is electrostatic potential
    holes = NA*np.exp(-p/VT)
    electrons = ni**2/NA*np.exp(p/VT)
    return q*(holes-electrons-NA)

for iteration in range(10000):

    old = psi.copy()

    for i in range(1,N-1):

        rho = charge(psi[i])

        psi[i] = 0.5*(psi[i+1]+psi[i-1]
                     +dx*dx*rho/eps_si)

    # gate boundary:
    # -eps_si*dpsi/dx = Cox(VG-VFB-psi_s)
    psi[0] = (psi[1]+dx*Cox/eps_si*(VG-VFB))/(1+dx*Cox/eps_si)

    # neutral bulk
    psi[-1] = 0

    if np.max(abs(psi-old)) < 1e-10:
        break

print("iterations =",iteration)
print("surface potential =",psi[0]," V")

plt.plot(x*1e4,psi)
plt.xlabel("Depth x (um)")
plt.ylabel("Potential psi (V)")
plt.grid()
plt.show()
