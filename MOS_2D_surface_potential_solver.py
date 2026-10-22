import numpy as np
import matplotlib.pyplot as plt

# 2-D MOS electrostatic surface-potential reconstruction
# No curve fitting parameters

q = 1.602e-19
eps0 = 8.854e-14
eps_si = 11.7*eps0
eps_ox = 3.9*eps0
VT = 0.02585

NA = 1e16
ND = 1e20
ni = 1e10
tox = 10e-7

phiB = VT*np.log(NA/ni)
Xdmax = np.sqrt(2*eps_si*(2*phiB)/(q*NA))

lam = np.sqrt((eps_si/eps_ox)*tox*Xdmax)

Vbi = VT*np.log(NA*ND/ni**2)
Vds = 1.5
Von = 0.30        # replace by self-consistent MOS gate solution

L = 0.5e-4
Xmax = 0.5e-4

Nx,Ny = 150,200
x = np.linspace(0,Xmax,Nx)
y = np.linspace(0,L,Ny)

phi_s = (Von
 +(Vbi-Von)*np.sinh((L-y)/lam)/np.sinh(L/lam)
 +(Vbi+Vds-Von)*np.sinh(y/lam)/np.sinh(L/lam))

# vertical extension
psi = np.zeros((Nx,Ny))
for i,xx in enumerate(x):
    psi[i,:] = phi_s*np.exp(-xx/Xdmax)

psi[-1,:] = 0

# plots
plt.figure()
plt.contourf(y*1e4,x*1e4,psi,40)
plt.xlabel("y (um)")
plt.ylabel("x (um)")
plt.colorbar(label="psi(V)")
plt.savefig("psi_xy.png",dpi=300)

plt.figure()
plt.plot(y/L,phi_s)
plt.xlabel("y/L")
plt.ylabel("surface potential (V)")
plt.grid()
plt.savefig("phi_surface.png",dpi=300)

from mpl_toolkits.mplot3d import Axes3D
Y,X=np.meshgrid(y*1e4,x*1e4)

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.plot_surface(Y,X,psi)
ax.set_xlabel("y (um)")
ax.set_ylabel("x (um)")
ax.set_zlabel("psi(V)")
plt.savefig("psi_3D.png",dpi=300)
