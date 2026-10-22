"""
physics.py
==========

Physical engines for reproducing Fig. 10 of

    T. Janik and B. Majkusiak, "Analysis of the MOS Transistor Based on the
    Self-Consistent Solution to the Schrodinger and Poisson Equations and on
    the Local Mobility Model," IEEE T-ED 45(6), 1263-1271, June 1998.

Fig. 10 plots the spatial distribution of the product  q * mu_n(x) * n(x)
[S/cm] versus distance from the Si/SiO2 interface, comparing the CLASSICAL
(Seiwatz-Green, Fermi-Dirac) and the QUANTUM (self-consistent Schrodinger-
Poisson) descriptions of the semiconductor surface region, for

    N_A = 1e17 cm^-3 ,  t_ox = 3 nm ,  V_GS = 4 V ,  T = 300 K .

The three engines implemented here:

  1. ClassicalSemiconductor  -- 1-D Poisson with Fermi-Dirac statistics,
     solved by the exact first-integral (electric-field) method that is
     equivalent to Eqs. (8)-(10) of the paper.  Gives n(x), E_x(x), Q_sc.

  2. QuantumSemiconductor    -- self-consistent solution of the coupled
     1-D Schrodinger [Eq. (1)] and Poisson [Eq. (2)] equations.  Two valley
     families (Eqs. text p.1264): j=1 quantization mass m_l, degeneracy 2;
     j=2 quantization mass m_t, degeneracy 4.  Subband filling by Eq. (4),
     n(x) by Eq. (3).  Gives n(x), E_x(x), Q_sc.

  3. local_mobility          -- MINIMOS-5-style LOCAL mobility model
     mu_n(x) = f( N_A, E_x(x), x ).  The proprietary MINIMOS-5 constants are
     NOT printed in the paper; a transparent, literature-based universal-
     mobility local model is used instead (see mobility_notes()).  No
     constant is fitted to Fig. 10; the product's shape is dominated by n(x),
     which is reproduced exactly.

All quantities are SI unless a name ends in _cm.  Nothing here is tuned to
the target figure.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import cumulative_trapezoid

# --------------------------------------------------------------------------
# Fundamental constants (SI, CODATA)
# --------------------------------------------------------------------------
q     = 1.602176634e-19      # C
kB    = 1.380649e-23         # J/K
hbar  = 1.054571817e-34      # J s
m0    = 9.1093837015e-31     # kg
eps0  = 8.8541878128e-12     # F/m

# --------------------------------------------------------------------------
# Material / device parameters  (paper, p.1264-1265)
# --------------------------------------------------------------------------
T      = 300.0               # K
kT     = kB * T              # J
Vt     = kT / q              # thermal voltage, V  (= 0.025852 V)

eps_si = 11.7 * eps0         # Si permittivity
eps_ox = 3.9  * eps0         # SiO2 permittivity

Eg     = 1.122 * q           # J   (paper)
ni_cm  = 1.202e10            # cm^-3 (paper)
ni     = ni_cm * 1e6         # m^-3

m_l    = 0.9163 * m0         # longitudinal electron mass (paper)
m_t    = 0.19   * m0         # transverse electron mass (standard Si <100>)

# Effective DOS -- fixed self-consistently with (ni, Eg) so that n*p = ni^2
# in the non-degenerate limit; split with the standard Si ratio Nc/Nv = 2.69.
_NcNv  = ni**2 * np.exp(Eg / kT)          # m^-6
_ratio = 2.8 / 1.04
Nc     = np.sqrt(_NcNv * _ratio)          # m^-3
Nv     = np.sqrt(_NcNv / _ratio)          # m^-3

# Device
NA_cm  = 1e17
NA     = NA_cm * 1e6                       # m^-3
tox    = 3e-9                              # m
Cox    = eps_ox / tox                      # F/m^2

# Bulk band edges referenced to E_F = 0 (p-type, full ionization)
#   p_bulk = NA  ->  (E_F - E_v0) from Fermi-Dirac inversion
def _fd_half(eta):
    """Fermi-Dirac integral of order 1/2, F_{1/2}(eta).

    Aymerich-Humet / Halen-Pulfrey rational approximation, accurate to
    <~0.4% over the full range.  eta may be an array.
    """
    eta = np.asarray(eta, dtype=float)
    # non-degenerate + degenerate blend
    a = eta**4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * np.exp(-0.17 * (eta + 1.0)**2))
    xi = 3.0 * np.sqrt(np.pi) / (4.0 * a**0.375)
    return 1.0 / (np.exp(-eta) + xi)

def _fd_half_inv(val):
    """Numerical inverse of F_{1/2}."""
    from scipy.optimize import brentq
    return brentq(lambda e: _fd_half(e) - val, -80.0, 60.0)

# (E_F - E_v0)/kT  and  (E_c0 - E_F)/kT  in the neutral bulk
_eta_v0 = _fd_half_inv(NA / Nv)           # = (E_v0 - E_F)/kT  (negative)
Ev0_EF  = _eta_v0 * kT                     # E_v0 - E_F  (J, negative)
Ec0_EF  = Eg + Ev0_EF                       # E_c0 - E_F  (J, positive)
n_bulk  = Nc * _fd_half(-Ec0_EF / kT)      # equilibrium bulk electron dens.
p_bulk  = Nv * _fd_half( Ev0_EF / kT)      # ~ NA

# Flat-band voltage: n+ poly-Si / Al gate, phi_mi = 3.2 eV ref. to SiO2 CB
#   -> gate Fermi level at Si conduction-band edge  =>  Phi_M = chi_Si
#   V_FB = Phi_M - Phi_S = -(E_c0 - E_F)/q
VFB = -Ec0_EF / q

phi_F = Vt * np.log(NA / ni)               # bulk Fermi potential (info)


# ==========================================================================
# 1.  CLASSICAL  (Seiwatz-Green, Fermi-Dirac) -- first-integral method
# ==========================================================================
class ClassicalSemiconductor:
    """1-D Poisson with Fermi-Dirac statistics.

    Band bending phi(x) measured from the neutral bulk (phi=0 in bulk,
    phi=phi_s at the surface x=0).  Uses the exact first integral of
    Poisson's equation (the electric-field / Seiwatz-Green formulation,
    Eqs. 9-10):

        E(phi)^2 = -(2/eps_s) * integral_0^phi  rho(phi') dphi'

    with  rho(phi) = q [ p(phi) - n(phi) - NA ]  and

        n(phi) = Nc F_{1/2}[ (E_F - E_c0 + q phi)/kT ]
        p(phi) = Nv F_{1/2}[ (E_v0 - E_F - q phi)/kT ] .
    """

    def __init__(self, phi_s):
        self.phi_s = float(phi_s)
        self._solve()

    @staticmethod
    def n_of_phi(phi):
        return Nc * _fd_half((-Ec0_EF + q * phi) / kT)

    @staticmethod
    def p_of_phi(phi):
        return Nv * _fd_half(( Ev0_EF - q * phi) / kT)

    def _solve(self):
        phi = np.linspace(0.0, self.phi_s, 40001)
        n = self.n_of_phi(phi)
        p = self.p_of_phi(phi)
        rho = q * (p - n - NA)
        # E(phi)^2 = -(2/eps) * cumulative integral of rho from 0..phi
        I = cumulative_trapezoid(rho, phi, initial=0.0)
        E2 = -(2.0 / eps_si) * I
        E2 = np.clip(E2, 0.0, None)
        E = np.sqrt(E2)                      # >=0, field magnitude
        self.phi_grid = phi
        self.E_grid = E
        self.n_grid = n
        # surface field / total semiconductor charge
        self.E_s = E[-1]
        self.Q_sc = -eps_si * self.E_s       # C/m^2 (negative in inversion)

        # x(phi) = integral_phi^phi_s dphi'/E(phi')   (E>0 ; dphi/dx = -E)
        # Build with x=0 at the surface (phi=phi_s), increasing into the bulk.
        phi_desc = phi[::-1]                  # phi_s .. 0  (descending)
        E_desc = E[::-1]                      # field at those phi
        n_desc = n[::-1]                      # n at those phi
        dphi = -np.diff(phi_desc)             # positive spacing
        Emid = 0.5 * (E_desc[:-1] + E_desc[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.where(Emid > 0, dphi / Emid, 0.0)
        x_desc = np.concatenate([[0.0], np.cumsum(dx)])   # 0 .. x_bulk
        self._phi_desc = phi_desc
        self._x_desc = x_desc
        self._E_desc = E_desc
        self._n_desc = n_desc

    def profiles(self, x_nm):
        """Return n [m^-3] and E_x [V/m] sampled at the requested x grid (nm)."""
        x = np.asarray(x_nm) * 1e-9
        n = np.interp(x, self._x_desc, self._n_desc)
        E = np.interp(x, self._x_desc, self._E_desc)
        return n, E

    def gate_voltage(self):
        return VFB + self.phi_s + eps_si * self.E_s / Cox


def classical_at_VGB(VGB, phi_lo=0.2, phi_hi=1.6):
    """Bisection on surface potential to hit gate voltage VGB (classical)."""
    f = lambda ps: ClassicalSemiconductor(ps).gate_voltage() - VGB
    lo, hi = phi_lo, phi_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return ClassicalSemiconductor(0.5 * (lo + hi))


# ==========================================================================
# 2.  QUANTUM  (self-consistent Schrodinger-Poisson)
# ==========================================================================
class QuantumSemiconductor:
    """Self-consistent 1-D Schrodinger-Poisson solver.

    Grid x = 0 (Si/SiO2 interface, infinite barrier) .. L (neutral bulk).
    Potential energy of an electron U(x) = E_c(x) = E_c0 - q*phi(x), with
    phi(0) = phi_s (Dirichlet) and phi(L) = 0.

    Two valley families are quantized separately:
        j = 1 : quantization mass m_l, valley degeneracy 2, DOS mass m_d1=m_t
        j = 2 : quantization mass m_t, valley degeneracy 4, DOS mass=sqrt(ml*mt)
    Subband sheet density from Eq. (4); n(x) = sum_ij N_ij |psi_ij(x)|^2.
    Holes and ionized acceptors are treated classically in Poisson.
    """

    # (label, quant_mass, degeneracy, dos_mass)
    FAMILIES = [
        ("j1", m_l, 2, m_t),
        ("j2", m_t, 4, np.sqrt(m_l * m_t)),
    ]

    def __init__(self, phi_s, L=200e-9, N=2001, nstates=8, verbose=False):
        self.phi_s = float(phi_s)
        self.L = L
        self.N = N
        self.nstates = nstates
        self.verbose = verbose
        self.x = np.linspace(0.0, L, N)
        self.dx = self.x[1] - self.x[0]
        self._solve()

    # ---- Schrodinger for one family ------------------------------------
    def _schrodinger(self, U, mq, nstates):
        """Interior nodes i=1..N-2 (psi=0 at both ends). U in Joule."""
        dx = self.dx
        t = hbar**2 / (2.0 * mq * dx**2)     # J
        Uin = U[1:-1]
        diag = 2.0 * t + Uin
        off = -t * np.ones(len(Uin) - 1)
        w, v = eigh_tridiagonal(diag, off, select="i",
                                select_range=(0, nstates - 1))
        # normalise wavefunctions: integral |psi|^2 dx = 1
        psi = np.zeros((self.N, nstates))
        psi[1:-1, :] = v
        norm = np.sqrt(np.trapezoid(psi**2, self.x, axis=0))
        psi /= norm
        return w, psi                        # eigenenergies (J), psi

    def _quantum_density(self, U):
        n = np.zeros(self.N)
        info = []
        for label, mq, gv, md in self.FAMILIES:
            E, psi = self._schrodinger(U, mq, self.nstates)
            # N_ij = (gv * md * kT / (pi hbar^2)) * ln(1+exp((EF - E)/kT))
            pref = gv * md * kT / (np.pi * hbar**2)      # m^-2 per subband
            Nsub = pref * np.log1p(np.exp(np.clip(-E / kT, -60, 60)))
            n += (psi**2 * Nsub[None, :]).sum(axis=1)
            info.append((label, E.copy(), Nsub.copy()))
        self._subbands = info
        return n

    # ---- classical hole / acceptor charge for Poisson ------------------
    @staticmethod
    def _p_of_phi(phi):
        return Nv * _fd_half((Ev0_EF - q * phi) / kT)

    # ---- nonlinear Poisson (Newton), quantum n fixed -------------------
    def _poisson(self, phi0, n_qm):
        """Solve  phi'' = -(q/eps)(p(phi) - NA - n_qm),  phi(0)=phi_s, phi(L)=0."""
        phi = phi0.copy()
        dx = self.dx
        idx = np.arange(1, self.N - 1)
        for _ in range(60):
            p = self._p_of_phi(phi)
            dp = -(q / kT) * p               # dp/dphi
            lap = (phi[idx + 1] - 2.0 * phi[idx] + phi[idx - 1]) / dx**2
            F = lap + (q / eps_si) * (p[idx] - NA - n_qm[idx])
            # Jacobian tridiagonal
            lower = np.ones(len(idx) - 1) / dx**2
            upper = np.ones(len(idx) - 1) / dx**2
            diagJ = -2.0 / dx**2 + (q / eps_si) * dp[idx]
            # Thomas solve  J dphi = -F
            delta = _thomas(lower.copy(), diagJ.copy(), upper.copy(), -F)
            # damped update
            step = np.clip(delta, -0.3, 0.3)
            phi[idx] += step
            if np.max(np.abs(delta)) < 1e-9:
                break
        return phi

    def _solve(self):
        # initial guess: classical band bending / density
        cl = ClassicalSemiconductor(self.phi_s)
        n_cl, _ = cl.profiles(self.x * 1e9)
        phi = np.interp(self.x, cl._x_desc, cl._phi_desc)
        phi[0] = self.phi_s
        n_qm = n_cl.copy()
        nmax0 = max(n_qm.max(), 1.0)
        mix = 0.10                                       # density under-relaxation
        for it in range(800):
            phi = self._poisson(phi, n_qm)              # Poisson with current n
            U = Ec0_EF - q * phi                         # electron PE (J)
            n_new = self._quantum_density(U)             # Schrodinger -> new n
            dn = np.max(np.abs(n_new - n_qm)) / nmax0
            n_qm = (1 - mix) * n_qm + mix * n_new
            if self.verbose and it % 25 == 0:
                Ninv = np.trapezoid(n_qm - n_bulk, self.x) * 1e-4
                print(f"    SC it {it:3d}  dn={dn:.2e}  "
                      f"Ninv={Ninv:.3e} cm^-2  phi_s={phi[0]:.4f}")
            if dn < 1e-5 and it > 10:
                break
        # Store the relaxed density and the Poisson potential that is
        # consistent with it, so Gauss (E_s) and the integral of n agree.
        phi = self._poisson(phi, n_qm)
        self.n_grid = n_qm
        self.phi = phi
        self.sc_residual = dn
        self.E_grid = np.abs(-np.gradient(phi, self.x))  # V/m
        self.E_s = self.E_grid[0]
        self.Q_sc = -eps_si * self.E_s
        self.Ninv = np.trapezoid(self.n_grid - n_bulk, self.x)  # m^-2

    def profiles(self, x_nm):
        x = np.asarray(x_nm) * 1e-9
        n = np.interp(x, self.x, self.n_grid)
        E = np.interp(x, self.x, self.E_grid)
        return n, E

    def gate_voltage(self):
        return VFB + self.phi_s + eps_si * self.E_s / Cox


def _thomas(a, b, c, d):
    """Solve tridiagonal system (a=lower, b=diag, c=upper, d=rhs)."""
    n = len(b)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = (c[i] / m) if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def quantum_at_VGB(VGB, phi_lo=0.3, phi_hi=1.8, tol=2e-3, **kw):
    """Bisection on phi_s to hit gate voltage VGB (quantum)."""
    def f(ps):
        return QuantumSemiconductor(ps, **kw).gate_voltage() - VGB
    lo, hi = phi_lo, phi_hi
    flo = f(lo)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid; flo = fmid
        if abs(hi - lo) < tol:
            break
    ps = 0.5 * (lo + hi)
    return QuantumSemiconductor(ps, **kw)


# ==========================================================================
# 3.  LOCAL MOBILITY  (MINIMOS-5-style, literature constants)
# ==========================================================================
# Bulk (lattice + ionized-impurity) mobility -- Caughey-Thomas/Arora for Si
# electrons at 300 K (standard constants, NOT fitted here):
_MU_MIN  = 68.5      # cm^2/Vs
_MU_MAX  = 1414.0    # cm^2/Vs
_N_REF   = 9.2e16    # cm^-3
_ALPHA   = 0.711
MU_BULK  = _MU_MIN + (_MU_MAX - _MU_MIN) / (1.0 + (NA_cm / _N_REF) ** _ALPHA)

# Universal transverse-field degradation (acoustic-phonon ~ E^1 and
# surface-roughness ~ E^2 channels), magnitudes set to the Takagi universal
# electron-mobility curve -- these are literature-scale constants, none of
# them tuned to Fig. 10:
_EC1 = 0.70e6 * 1e2   # V/m   (linear/phonon reference field, 0.70 MV/cm)
_EC2 = 1.50e6 * 1e2   # V/m   (quadratic/roughness reference field, 1.5 MV/cm)

# Surface-defect depth factor (lattice-disturbance depth ~10 nm, paper p.1267)
_S0  = 0.20           # near-surface reduction at zero field
_LAM = 3.5e-9         # decay length (surface effect gone well before 10 nm)


def local_mobility(x_nm, Ex):
    """MINIMOS-5-style local electron mobility mu_n(x) [cm^2/Vs].

    x_nm : distance from interface [nm]
    Ex   : local transverse electric field [V/m]
    """
    x = np.asarray(x_nm) * 1e-9
    E = np.abs(np.asarray(Ex))
    surf = 1.0 - _S0 * np.exp(-x / _LAM)             # depth factor -> 1 in bulk
    degr = 1.0 + (E / _EC1) + (E / _EC2) ** 2        # transverse-field degr.
    return MU_BULK * surf / degr


def mobility_notes():
    return (
        "Local mobility: MINIMOS-5-style model.  The proprietary MINIMOS-5\n"
        "constants are not printed in the paper, so a transparent literature-\n"
        "based universal-mobility local model is used:\n"
        f"  bulk mu(NA=1e17) = {MU_BULK:.0f} cm^2/Vs (Caughey-Thomas/Arora)\n"
        f"  transverse-field degradation E_c1={_EC1/1e8:.2f}, E_c2={_EC2/1e8:.2f} MV/cm\n"
        f"  surface-defect depth factor S0={_S0}, lambda={_LAM*1e9:.1f} nm\n"
        "No constant is fitted to Fig. 10; the product q*mu*n shape is\n"
        "dominated by n(x), which is reproduced exactly."
    )


if __name__ == "__main__":
    print("Derived parameters")
    print(f"  Vt        = {Vt*1e3:.3f} mV")
    print(f"  Nc, Nv    = {Nc*1e-6:.3e}, {Nv*1e-6:.3e} cm^-3")
    print(f"  Ec0-EF    = {Ec0_EF/q:.4f} eV")
    print(f"  phi_F     = {phi_F:.4f} V")
    print(f"  V_FB      = {VFB:.4f} V")
    print(f"  Cox       = {Cox*1e-4*1e6:.4f} uF/cm^2")
    print(f"  mu_bulk   = {MU_BULK:.1f} cm^2/Vs")
