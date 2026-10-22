"""1-D self-consistent Schrodinger-Poisson solver for an nMOS inversion layer.

The script is intended to reproduce the *physics and shape* of Spinelli et al.
Fig. 9 without digitizing that figure. It separately computes:
  1) a classical Poisson-Boltzmann solution, and
  2) a quantum Schrodinger-Poisson solution using finite differences.

Because the paper does not publish the complete channel doping profile,
flat-band voltage/work-function difference, interface charge, and the exact
local gate bias for Fig. 9, the numerical curve is Fig.-9-like rather than an
exact reproduction. Vfb is therefore exposed as a user parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, e, epsilon_0, k
from scipy.linalg import eigh
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import factorized

# ----------------------- physical / device parameters -----------------------
T = 300.0
VT = k*T/e
kT = k*T
q = e

NA_cm3 = 2.5e18             # highest peak channel doping quoted in the paper
ni_cm3 = 1.0e10
Nc_cm3 = 2.8e19
NA = NA_cm3*1e6             # cm^-3 -> m^-3
ni = ni_cm3*1e6
Nc = Nc_cm3*1e6
n_bulk = ni**2/NA

# Energy of the bulk conduction-band edge measured from EF.
# From n_bulk = Nc exp[-(Ec-EF)/kT].
Ec_bulk_minus_EF_eV = VT*np.log(Nc/n_bulk)

eps_si = 11.7*epsilon_0
eps_ox = 3.9*epsilon_0
tox = 10e-9                 # 100 Angstrom
VG = 5.0
Vfb = -1.0                  # illustrative; paper does not specify it for Fig. 9

L = 40e-9                   # paper states a 40 nm QM domain
dz = 0.2e-9                 # paper states 0.2 nm near-interface spacing
z = np.arange(0.0, L + 0.5*dz, dz)
N = z.size

# (100) Si valley classes: (valley degeneracy, confinement mass, in-plane DOS mass)
# The 2-fold Delta2 valleys have heavy confinement mass; the 4-fold Delta4 valleys
# have light confinement mass. The spin factor is already included in m/(pi hbar^2).
valleys = [
    (2, 0.916, 0.190),
    (4, 0.190, np.sqrt(0.916*0.190)),
]

# ------------------------------- Poisson matrix -----------------------------
# Poisson: d^2 phi/dz^2 = -rho/eps_si
# Surface Robin BC from oxide displacement continuity:
#   dphi/dz|0 = (eps_ox/(eps_si tox)) [phi(0) - (VG-Vfb)]
# Bulk BC: phi(L)=0.
Cox_bc = eps_ox/(eps_si*tox)
A = lil_matrix((N, N), dtype=float)
A[0, 0] = -(1.0/dz + Cox_bc)
A[0, 1] = 1.0/dz
for i in range(1, N-1):
    A[i, i-1] = 1.0/dz**2
    A[i, i]   = -2.0/dz**2
    A[i, i+1] = 1.0/dz**2
A[-1, -1] = 1.0
A = csc_matrix(A)
solve_poisson_linear = factorized(A)


def poisson_update(n_m3, phi_old):
    """One Gummel Poisson update for a specified electron density."""
    # Nondegenerate holes are sufficient here; p_bulk ~ NA.
    p = NA*np.exp(np.clip(-phi_old/VT, -100.0, 100.0))
    rho = q*(p - n_m3 - NA)
    b = np.zeros(N)
    b[0] = -Cox_bc*(VG - Vfb)
    b[1:-1] = -rho[1:-1]/eps_si
    b[-1] = 0.0
    return solve_poisson_linear(b)


# -------------------------- classical self-consistency ----------------------
def solve_classical(alpha=0.005, tol=1e-7, max_iter=5000):
    phi = np.linspace(0.8, 0.0, N)
    for it in range(max_iter):
        n = n_bulk*np.exp(np.clip(phi/VT, -100.0, 60.0))
        phi_raw = poisson_update(n, phi)
        err = np.max(np.abs(phi_raw - phi))
        phi = (1.0-alpha)*phi + alpha*phi_raw
        if err < tol:
            break
    n = n_bulk*np.exp(np.clip(phi/VT, -100.0, 60.0))
    return phi, n, it+1


# ------------------------- Schrodinger finite difference --------------------
def quantum_density(phi, return_levels=False):
    """Return n_QM(z) by solving the finite-difference Hamiltonian.

    Dirichlet wavefunction BCs psi(0)=psi(L)=0 are imposed by solving only on
    interior nodes. Every finite-box eigenstate is included. This is a clean
    fully numerical alternative to the paper's 3-subband + classical-tail
    acceleration.
    """
    M = N - 2
    Ec = q*(Ec_bulk_minus_EF_eV - phi[1:-1])  # joule, with EF=0
    n_int = np.zeros(M)
    level_summary = []

    for gv, mz_rel, md_rel in valleys:
        mz = mz_rel*m_e
        md = md_rel*m_e
        t = hbar**2/(2.0*mz*dz**2)

        # Tridiagonal Hamiltonian:
        # H_ii   = Ec_i + hbar^2/(m_z dz^2)
        # H_i,i+1=H_i+1,i = -hbar^2/(2 m_z dz^2)
        H = np.diag(Ec + 2.0*t)
        H += np.diag(np.full(M-1, -t), 1)
        H += np.diag(np.full(M-1, -t), -1)

        E, U = eigh(H, check_finite=False)

        # eigh normalizes sum |U_i|^2=1. Convert to continuum normalization
        # sum |psi_i|^2 dz = 1.
        psi = U/np.sqrt(dz)

        # 2-D Fermi-Dirac sheet population of each subband:
        # Ns = gv*md*kT/(pi hbar^2) ln[1+exp((EF-E)/kT)], EF=0.
        x = -E/kT
        logF = np.where(x > 50.0, x, np.log1p(np.exp(np.clip(x, -700.0, 50.0))))
        Ns = gv*md*kT/(np.pi*hbar**2)*logF   # m^-2

        n_int += (psi**2) @ Ns               # m^-3
        if return_levels:
            level_summary.append((gv, E[:6]/q, Ns[:6]/1e4))  # eV, cm^-2

    n = np.zeros(N)
    n[1:-1] = n_int
    if return_levels:
        return n, level_summary
    return n


def solve_quantum(alpha=0.03, tol=1e-6, max_iter=600):
    # Start from a moderate band-bending profile rather than the classical
    # strong-inversion solution; under-relaxation provides stable Gummel steps.
    phi = np.linspace(1.0, 0.0, N)
    for it in range(max_iter):
        n = quantum_density(phi)
        phi_raw = poisson_update(n, phi)
        err = np.max(np.abs(phi_raw - phi))
        phi = (1.0-alpha)*phi + alpha*phi_raw
        if err < tol:
            break
    n, levels = quantum_density(phi, return_levels=True)
    return phi, n, levels, it+1


if __name__ == "__main__":
    phi_cl, n_cl, it_cl = solve_classical()
    phi_qm, n_qm, levels, it_qm = solve_quantum()

    print(f"Classical iterations: {it_cl}")
    print(f"Quantum iterations:   {it_qm}")
    print(f"Classical surface potential = {phi_cl[0]:.6f} V")
    print(f"Quantum surface potential   = {phi_qm[0]:.6f} V")
    print(f"QM density peak at z = {z[np.argmax(n_qm)]*1e9:.3f} nm")
    print(f"max n_classical = {np.max(n_cl)/1e6:.3e} cm^-3")
    print(f"max n_QM        = {np.max(n_qm)/1e6:.3e} cm^-3")
    for j,(gv,E6,N6) in enumerate(levels, start=1):
        print(f"Valley class {j}, degeneracy {gv}: first six E-EF [eV] =", E6)

    out = np.column_stack((z*1e9, phi_cl, phi_qm, n_cl/1e6, n_qm/1e6))
    np.savetxt(
        "/mnt/data/fig9_SP_numerical.csv", out, delimiter=",",
        header="z_nm,phi_classical_V,phi_quantum_V,n_classical_cm-3,n_quantum_cm-3",
        comments=""
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogy(z*1e9, n_qm/1e6, label="Quantum solution")
    ax.semilogy(z*1e9, n_cl/1e6, "--", label="Classical solution")
    ax.set_xlim(0, 25)
    ax.set_ylim(1, 1e21)
    ax.set_xlabel("Distance from interface [nm]")
    ax.set_ylabel("Electron concentration [cm$^{-3}$]")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)

    ins = ax.inset_axes([0.55, 0.55, 0.40, 0.36])
    ins.plot(z*1e9, n_qm/1e6, label="QM")
    ins.plot(z*1e9, n_cl/1e6, "--", label="Classical")
    ins.set_xlim(0, 4)
    ymax = 1.08*max(np.max(n_qm), np.max(n_cl))/1e6
    ins.set_ylim(0, ymax)
    ins.set_xlabel("z [nm]", fontsize=8)
    ins.set_ylabel("n [cm$^{-3}$]", fontsize=8)
    ins.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig("/mnt/data/Fig9_SP_from_first_principles.png", dpi=220)
    print("Wrote /mnt/data/Fig9_SP_from_first_principles.png")
    print("Wrote /mnt/data/fig9_SP_numerical.csv")
