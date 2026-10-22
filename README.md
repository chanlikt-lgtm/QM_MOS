# QM_MOS — Reproduction of Fig. 10 of Janik & Majkusiak (1998)

Numerical reproduction of **Fig. 10** of

> T. Janik and B. Majkusiak, *"Analysis of the MOS Transistor Based on the
> Self-Consistent Solution to the Schrödinger and Poisson Equations and on the
> Local Mobility Model,"* IEEE Trans. Electron Devices, **45**(6), 1263–1271,
> June 1998.

Fig. 10 is the spatial distribution of the product **q·n(x)·μₙ(x)** [S/cm]
versus distance from the Si/SiO₂ interface, comparing the **classical** and
**quantum** descriptions of the semiconductor surface region, for

> Nₐ = 10¹⁷ cm⁻³, t_ox = 3 nm, V_GS = 4 V, T = 300 K.

No parameter is fitted to the target figure.

## Method

Three physics engines (`fig10_reproduction/physics.py`):

1. **Classical** — 1-D Poisson with Fermi–Dirac statistics, solved by the exact
   electric-field first integral (Seiwatz–Green, Eqs. 8–10).
2. **Quantum** — self-consistent solution of the coupled 1-D Schrödinger (Eq. 1)
   and Poisson (Eq. 2) equations over two valley families
   (j=1: m_x=m_ℓ, deg. 2; j=2: m_x=m_t, deg. 4), subband filling by Eq. (4),
   n(x) = Σ N_ij|ψ_ij|² (Eq. 3). Surface potential fixed by the MOS gate
   equation to V_GB = 4 V.
3. **Local mobility** — MINIMOS-5-style. The proprietary MINIMOS-5 constants are
   not printed in the paper, so a transparent, literature-based
   universal-mobility local model is used (every constant declared; none fitted).

## Validation (V_GS = 4 V)

| quantity | classical | quantum | paper |
|---|---|---|---|
| φ_s [V] | 1.165 | 1.477 | ~1.18 / higher (Fig. 6) |
| N_inv [cm⁻²] | 2.64e13 | 2.41e13 | ~2.3e13 (Fig. 7) |
| peak n [cm⁻³] | 7.38e20 | 1.80e20 | ~1.8e20 (Fig. 3) |
| peak q·n·μ [S/cm] @ nm | 4737 @ 0.14 | 2599 @ 1.12 | ~3.4e3 @ 0.5 / ~2.4e3 @ 1.3 |

The quantum curve matches the paper closely. The classical near-surface
magnitude runs ~30–40% high (weaker near-surface mobility roll-off than
MINIMOS-5); the product's shape and the quantum-vs-classical comparison — the
subject of Fig. 10 — are dominated by n(x), reproduced from first principles.
Independent cross-checks (peak n, N_inv ordering and magnitude, band-bending
trend) confirm the charge solution.

## Usage

```bash
pip install numpy scipy matplotlib
cd fig10_reproduction
python run_fig10.py      # -> fig10_reproduction.png, fig10_data.csv + summary
python make_report.py    # -> side-by-side PNG + PDF report in parent folder
```

The LaTeX report (`Fig10_reproduction_report.tex`) compiles with `pdflatex`
(run twice) to `Fig10_reproduction_report.pdf`.

## Files

```
Fig10_reproduction_report.tex / .pdf   LaTeX report (nomenclature + method + results)
fig10_reproduction/
  physics.py          classical + quantum + mobility engines
  run_fig10.py        driver (solves both models at V_GB=4 V)
  make_report.py      side-by-side figure + multi-page PDF report
  fig10_reproduction.png   product q·n·μ and underlying n(x), μ(x)
  fig10_side_by_side.png   original Fig. 10 vs this work
  fig10_data.csv      x, n, μ, q·n·μ for both models
```

> The copyrighted IEEE source paper is **not** included in this repository.
