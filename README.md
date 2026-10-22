# QM_MOS: Quantum-Mechanical (QM) Self-Consistent Schrödinger-Poisson & Semiclassical MOS Solvers

This repository hosts reproducible numerical solvers, benchmark reproductions, and technical derivations for **quantum-mechanical (QM) confinement effects and self-consistent Schrödinger-Poisson electrostatics in MOS / MOSFET inversion layers**.

---

## 1. Reproduction of Fig. 10 of Janik & Majkusiak (1998)

Numerical reproduction of **Fig. 10** of:

> T. Janik and B. Majkusiak, *"Analysis of the MOS Transistor Based on the Self-Consistent Solution to the Schrödinger and Poisson Equations and on the Local Mobility Model,"* **IEEE Trans. Electron Devices**, 45(6), 1263–1271, June 1998.

Fig. 10 is the spatial distribution of the product **$q \cdot n(x) \cdot \mu_n(x)$** [$\mathrm{S/cm}$] versus distance from the $\mathrm{Si/SiO_2}$ interface, comparing the **classical** and **quantum** descriptions of the semiconductor surface region, for:
> $N_A = 10^{17}\,\mathrm{cm^{-3}}$, $t_{\mathrm{ox}} = 3\,\mathrm{nm}$, $V_{\mathrm{GS}} = 4\,\mathrm{V}$, $T = 300\,\mathrm{K}$.

No parameter is fitted to the target figure.

### Method
Three physics engines (`fig10_reproduction/physics.py`):
1. **Classical:** 1-D Poisson with Fermi–Dirac statistics, solved by the exact electric-field first integral (Seiwatz–Green, Eqs. 8–10).
2. **Quantum:** self-consistent solution of the coupled 1-D Schrödinger (Eq. 1) and Poisson (Eq. 2) equations over two valley families ($j=1$: $m_x=m_l$, deg. 2; $j=2$: $m_x=m_t$, deg. 4), subband filling by Eq. (4), $n(x) = \sum N_{ij} |\psi_{ij}|^2$ (Eq. 3). Surface potential fixed by the MOS gate equation to $V_{\mathrm{GB}} = 4\,\mathrm{V}$.
3. **Local mobility:** MINIMOS-5-style universal-mobility local model.

### Validation ($V_{\mathrm{GS}} = 4\,\mathrm{V}$)

| Quantity | Classical | Quantum | Paper Reference |
| :--- | :--- | :--- | :--- |
| $\phi_s$ [V] | 1.165 | 1.477 | ~1.18 / higher (Fig. 6) |
| $N_{\mathrm{inv}}$ [$\mathrm{cm^{-2}}$] | $2.64 \times 10^{13}$ | $2.41 \times 10^{13}$ | $\sim 2.3 \times 10^{13}$ (Fig. 7) |
| Peak $n$ [$\mathrm{cm^{-3}}$] | $7.38 \times 10^{20}$ | $1.80 \times 10^{20}$ | $\sim 1.8 \times 10^{20}$ (Fig. 3) |
| Peak $q \cdot n \cdot \mu$ [S/cm] @ nm | 4737 @ 0.14 | 2599 @ 1.12 | ~3.4e3 @ 0.5 / ~2.4e3 @ 1.3 |

---

## 2. Extended Schrödinger-Poisson & Quantum Onset Mapping

In advanced MOS nodes, quantum confinement causes wavefunction suppression at the oxide interface ($|\psi(0)|^2 = 0$), shifting the inversion centroid by $\Delta z_{\mathrm{qm}} \approx 1\text{--}2\,\mathrm{nm}$ and reducing the effective gate capacitance:
$$\frac{1}{C_{\mathrm{eff}}} = \frac{1}{C_{\mathrm{ox}}} + \frac{1}{C_{\mathrm{qm}}} = \frac{t_{\mathrm{ox}}}{\varepsilon_{\mathrm{ox}}} + \frac{\Delta z_{\mathrm{qm}}}{\varepsilon_{\mathrm{si}}}$$

### 1D Schrödinger-Poisson System
For (100) Silicon with longitudinal ($g_L = 2, m_l = 0.9163\,m_0, m_d = 0.2245\,m_0$) and transverse ($g_T = 4, m_t = 0.2245\,m_0, m_d = 0.4535\,m_0$) valleys:
$$-\frac{\hbar^2}{2 m_{z,v}} \frac{d^2 \psi_{i,v}(z)}{dz^2} + [E_c(z) - E_F] \psi_{i,v}(z) = E_{i,v} \psi_{i,v}(z)$$
$$N_{i,v} = \frac{g_v m_{d,v} k_B T}{\pi \hbar^2} \ln\left(1 + \exp\left[\frac{E_F - E_{i,v}}{k_B T}\right]\right)$$
$$n_{\mathrm{qm}}(z) = \sum_{v \in \{L, T\}} \sum_{i} N_{i,v} |\psi_{i,v}(z)|^2$$

Coupled nonlinearly with Poisson's equation:
$$\frac{d}{dz}\left(\varepsilon_{\mathrm{si}} \frac{d\phi}{dz}\right) = -q [p(z) - n_{\mathrm{qm}}(z) - N_A^-(z)]$$

---

## 3. Repository Structure

```
QM_MOS/
├── Fig10_reproduction_report.tex / .pdf   LaTeX report for Janik & Majkusiak (1998)
├── fig10_reproduction/
│   ├── physics.py                        Classical + Quantum + Mobility engines
│   ├── run_fig10.py                      Fig. 10 driver (solves both models at V_GB=4 V)
│   ├── make_report.py                    Report & side-by-side comparison generator
│   ├── fig10_reproduction.png            q*n*mu product, n(x), and mu(x)
│   ├── fig10_side_by_side.png            Original Fig. 10 vs reproduction
│   └── fig10_data.csv                    Tabulated spatial profile data
├── qm_onset_map.py                       2D/3D parameter scan mapping QM onset in MOS
├── schrodinger_poisson_fig9.py           1D Schrödinger-Poisson vs classical (Spinelli et al.)
├── MOS_1D_Poisson_Boltzmann_no_fit.py    Closed-form 1D MOS Poisson-Boltzmann baseline
├── MOS_2D_surface_potential_solver.py    2D channel surface potential reconstruction
├── poisson_efield_mos2d.py               2D electrostatic Poisson solver
├── fd_half_table.npz                     Precomputed Fermi-Dirac F_1/2 and F_-1/2 tables
└── documentation/
    ├── full_nonlinear_schrodinger_poisson_qm_onset_maps.pdf
    ├── full_nonlinear_schrodinger_poisson_3subband_compare.pdf
    ├── qm_current_density.pdf
    └── MOS_1D_Poisson_Boltzmann_appendix.pdf
```

---

## 4. Usage & Quickstart

```bash
# 1. Run Janik & Majkusiak (1998) Fig. 10 reproduction:
cd fig10_reproduction
python run_fig10.py
python make_report.py

# 2. Run 1D Schrödinger-Poisson solver (Spinelli et al. benchmark):
python schrodinger_poisson_fig9.py

# 3. Run classical parameter-free baselines:
python MOS_1D_Poisson_Boltzmann_no_fit.py
python MOS_2D_surface_potential_solver.py
```

---

## 5. References
1. T. Janik and B. Majkusiak, *"Analysis of the MOS Transistor Based on the Self-Consistent Solution to the Schrödinger and Poisson Equations and on the Local Mobility Model,"* **IEEE Trans. Electron Devices**, vol. 45, no. 6, pp. 1263–1271, Jun. 1998.
2. A. Spinelli, A. Pacelli, and A. L. Lacaita, *"Self-consistent 2-D model for quantum effects in n-MOS transistors,"* **IEEE Trans. Electron Devices**, vol. 47, no. 12, pp. 2366–2371, Dec. 2000.
3. F. Stern, *"Self-consistent results for n-type Si inversion layers,"* **Phys. Rev. B**, vol. 5, no. 12, pp. 4891–4899, Jun. 1972.
