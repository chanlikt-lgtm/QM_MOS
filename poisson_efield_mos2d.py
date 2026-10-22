import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# 2-D electrostatic Poisson solver for the sketched Si/SiO2 structure.
#
# Governing equations:
#     div(epsilon * grad(V)) = -rho
#     E = -grad(V)
#
# Assumed bias:
#     source p+  = 0 V
#     gate       = 0 V
#     drain p+   = -10 V
#     back Si    = 0 V
#
# Replace the assumed geometry and doping below with your real device values.

eps0 = 8.8541878128e-12
eps_si = 11.7 * eps0
eps_ox = 3.9 * eps0
q = 1.602176634e-19

# Geometry
Lx = 20e-6
si_depth = 5e-6
tox = 0.30e-6

Nx = 301
Ny = 107
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(-si_depth, tox, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Material map
epsilon = np.where(Y <= 0.0, eps_si, eps_ox)

# Example depletion/fixed-charge approximation:
# n-type body ND = 1e14 cm^-3 = 1e20 m^-3
ND = 1e20
rho = np.where(Y <= 0.0, q * ND, 0.0)

# Dirichlet masks
fixed = np.zeros((Ny, Nx), dtype=bool)
Vfix = np.zeros((Ny, Nx), dtype=float)

# Backside Si = 0 V
fixed[0, :] = True
Vfix[0, :] = 0.0

j_surface = np.argmin(np.abs(y))

# Assumed electrode spans
source_mask = (x >= 1.5e-6) & (x <= 4.5e-6)
gate_mask = (x >= 7.0e-6) & (x <= 12.0e-6)
drain_mask = (x >= 15.5e-6) & (x <= 18.5e-6)

fixed[j_surface, source_mask] = True
Vfix[j_surface, source_mask] = 0.0

fixed[j_surface, drain_mask] = True
Vfix[j_surface, drain_mask] = -10.0

fixed[-1, gate_mask] = True
Vfix[-1, gate_mask] = 0.0

# Unspecified outer boundaries are zero-normal-flux Neumann boundaries.

def idx(j, i):
    return j * Nx + i

def hmean(a, b):
    return 2.0 * a * b / (a + b)

# Solve -div(epsilon grad V) = rho
N = Nx * Ny
rows, cols, data = [], [], []
b = np.zeros(N)

for j in range(Ny):
    for i in range(Nx):
        k = idx(j, i)

        if fixed[j, i]:
            rows.append(k)
            cols.append(k)
            data.append(1.0)
            b[k] = Vfix[j, i]
            continue

        diag = 0.0

        neighbors = (
            (j, i - 1, dx * dx),
            (j, i + 1, dx * dx),
            (j - 1, i, dy * dy),
            (j + 1, i, dy * dy),
        )

        for jj, ii, dsq in neighbors:
            if 0 <= ii < Nx and 0 <= jj < Ny:
                c = hmean(epsilon[j, i], epsilon[jj, ii]) / dsq
                diag += c

                if fixed[jj, ii]:
                    b[k] += c * Vfix[jj, ii]
                else:
                    rows.append(k)
                    cols.append(idx(jj, ii))
                    data.append(-c)

        rows.append(k)
        cols.append(k)
        data.append(diag)
        b[k] += rho[j, i]

A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
V = spla.spsolve(A, b).reshape(Ny, Nx)

# Electric field
dVdy, dVdx = np.gradient(V, dy, dx)
Ex = -dVdx
Ey = -dVdy
Emag = np.hypot(Ex, Ey)

jmax, imax = np.unravel_index(np.argmax(Emag), Emag.shape)
print("Potential range:", V.min(), "to", V.max(), "V")
print("Raw peak |E|:", Emag[jmax, imax], "V/m")
print("Peak location:",
      x[imax] * 1e6, "um,",
      y[jmax] * 1e6, "um")
print("Raw peak |E|:", Emag[jmax, imax] / 1e8, "MV/cm")
print("Sharp ideal contact edges produce mesh-dependent local field peaks.")

# Potential + field direction
plt.figure(figsize=(10, 4.5))
levels = np.linspace(V.min(), V.max(), 40)
cf = plt.contourf(X * 1e6, Y * 1e6, V, levels=levels)
plt.colorbar(cf, label="Potential V (V)")
sy = slice(0, Ny, 5)
sx = slice(0, Nx, 10)
plt.quiver(X[sy, sx] * 1e6,
           Y[sy, sx] * 1e6,
           Ex[sy, sx],
           Ey[sy, sx])
plt.axhline(0.0)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("2-D Poisson solution: potential and electric-field direction")
plt.tight_layout()
plt.savefig("poisson_potential_field.png", dpi=180)
plt.show()

# Field magnitude
plt.figure(figsize=(10, 4.5))
cf2 = plt.contourf(X * 1e6, Y * 1e6, Emag / 1e6, levels=40)
plt.colorbar(cf2, label="|E| (MV/m)")
plt.axhline(0.0)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("Electric-field magnitude")
plt.tight_layout()
plt.savefig("poisson_efield_magnitude.png", dpi=180)
plt.show()
