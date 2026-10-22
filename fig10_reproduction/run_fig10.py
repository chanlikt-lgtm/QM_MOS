"""
run_fig10.py
============

Reproduce Fig. 10 of Janik & Majkusiak (IEEE T-ED, 1998):

    spatial distribution of  q * mu_n(x) * n(x)  [S/cm]  vs distance [nm],
    classical vs quantum descriptions, for
    N_A = 1e17 cm^-3, t_ox = 3 nm, V_GS = 4 V, T = 300 K.

Produces:
    fig10_reproduction.png   -- side-by-side replica of Fig. 10
    fig10_data.csv           -- x, n, mu, q*mu*n for both models
and prints a validation summary to stdout.

No parameter is fitted to the target figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import physics as P

VGB = 4.0
XMAX_NM = 8.0
NPTS = 801

# --------------------------------------------------------------------------
def solve_quantum_at_VGB(target, lo=1.30, hi=1.60):
    """Bisection on quantum surface potential to hit gate voltage `target`."""
    def g(ps):
        return P.QuantumSemiconductor(ps, L=200e-9, N=1601, nstates=6)
    flo = g(lo).gate_voltage() - target
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        qm = g(mid)
        fmid = qm.gate_voltage() - target
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid; flo = fmid
        if abs(hi - lo) < 1e-3:
            break
    return P.QuantumSemiconductor(0.5 * (lo + hi), L=200e-9, N=1601, nstates=6)


def main():
    x_nm = np.linspace(1e-3, XMAX_NM, NPTS)

    print("Solving CLASSICAL (Seiwatz-Green, Fermi-Dirac) ...")
    cl = P.classical_at_VGB(VGB)
    n_cl, E_cl = cl.profiles(x_nm)
    mu_cl = P.local_mobility(x_nm, E_cl)
    prod_cl = P.q * mu_cl * (n_cl * 1e-6)          # S/cm

    print("Solving QUANTUM (self-consistent Schrodinger-Poisson) ...")
    qm = solve_quantum_at_VGB(VGB)
    n_qm, E_qm = qm.profiles(x_nm)
    mu_qm = P.local_mobility(x_nm, E_qm)
    prod_qm = P.q * mu_qm * (n_qm * 1e-6)          # S/cm

    # ---- validation summary --------------------------------------------
    Ninv_cl = np.trapezoid(np.maximum(cl._n_desc - P.n_bulk, 0), cl._x_desc) * 1e-4
    print("\n" + "=" * 66)
    print("VALIDATION SUMMARY  (N_A=1e17 cm^-3, t_ox=3 nm, V_GS=4 V)")
    print("=" * 66)
    print(f"{'quantity':32s}{'classical':>16s}{'quantum':>16s}")
    print(f"{'surface potential phi_s [V]':32s}{cl.phi_s:>16.4f}{qm.phi_s:>16.4f}")
    print(f"{'gate voltage V_GB [V]':32s}{cl.gate_voltage():>16.4f}"
          f"{qm.gate_voltage():>16.4f}")
    print(f"{'inversion N_inv [cm^-2]':32s}{Ninv_cl:>16.3e}{qm.Ninv*1e-4:>16.3e}")
    print(f"{'surface field E_s [MV/cm]':32s}{cl.E_s*1e-8:>16.3f}{qm.E_s*1e-8:>16.3f}")
    ipk_c = np.argmax(prod_cl); ipk_q = np.argmax(prod_qm)
    print(f"{'peak q*mu*n [S/cm]':32s}{prod_cl[ipk_c]:>16.1f}{prod_qm[ipk_q]:>16.1f}")
    print(f"{'peak position [nm]':32s}{x_nm[ipk_c]:>16.2f}{x_nm[ipk_q]:>16.2f}")
    print(f"{'peak n [cm^-3]':32s}{n_cl.max()*1e-6:>16.3e}{n_qm.max()*1e-6:>16.3e}")
    print("=" * 66)
    print("\nReference (paper): quantum N_inv ~ 2.3e13 cm^-2 (Fig.7);")
    print("Fig.10 classical peak ~3.6e3 S/cm near ~0.3 nm, quantum peak")
    print("~2.3e3 S/cm near ~1.3 nm.\n")
    print(P.mobility_notes())

    # ---- save data ------------------------------------------------------
    np.savetxt(
        "fig10_data.csv",
        np.column_stack([x_nm, n_cl * 1e-6, mu_cl, prod_cl,
                         n_qm * 1e-6, mu_qm, prod_qm]),
        delimiter=",",
        header="x_nm,n_classical_cm-3,mu_classical_cm2Vs,qmun_classical_Scm,"
               "n_quantum_cm-3,mu_quantum_cm2Vs,qmun_quantum_Scm",
        comments="",
    )

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    ax.plot(x_nm, prod_qm, "b-", lw=2.0, label="quantum")
    ax.plot(x_nm, prod_cl, "r--", lw=2.0, label="classical")
    ax.set_xlabel("DISTANCE  [ nm ]")
    ax.set_ylabel(r"q $\mu_n$ n   [ S/cm ]")
    ax.set_xlim(0, XMAX_NM)
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    ax.set_title("Reproduction of Fig. 10")
    ax.text(0.55, 0.72,
            r"$N_A = 10^{17}\ \mathrm{cm^{-3}}$" + "\n"
            r"$t_{ox} = 3$ nm" + "\n" + r"$V_{GS} = 4$ V",
            transform=ax.transAxes, fontsize=11)

    ax = axes[1]
    ax.semilogy(x_nm, n_qm * 1e-6, "b-", lw=2.0, label="quantum n")
    ax.semilogy(x_nm, n_cl * 1e-6, "r--", lw=2.0, label="classical n")
    ax2 = ax.twinx()
    ax2.plot(x_nm, mu_qm, "b:", lw=1.5)
    ax2.plot(x_nm, mu_cl, "r:", lw=1.5)
    ax.set_xlabel("DISTANCE  [ nm ]")
    ax.set_ylabel(r"electron concentration  [ cm$^{-3}$ ]")
    ax2.set_ylabel(r"local mobility $\mu_n$  [ cm$^2$/Vs ]  (dotted)")
    ax.set_xlim(0, XMAX_NM)
    ax.set_ylim(1e16, 1e21)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Underlying n(x) and $\\mu_n$(x)")

    fig.tight_layout()
    fig.savefig("fig10_reproduction.png", dpi=150)
    print("wrote fig10_reproduction.png and fig10_data.csv")


if __name__ == "__main__":
    main()
