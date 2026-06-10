# FRC finite-zeta — validation & visualization

Reproduces every computational claim and every figure in
*"Finite Zeta, Phase Amplitudes, and the Emergence of Prime Distribution in the Finite Ring Continuum."*

## Files
- `frc_core.py` — arithmetic primitives (Möbius/totient sieves, Ramanujan/units-chart sums `c_q(n)` in both Kluyver and trigonometric form, discrete-log latitude tables, von Mangoldt and its convolution-inverse form, the units-chart resonance `R_L` (limit `(phi(n)/n)*Lambda(n)`, exact `Lambda=mu*log` after radical rescaling), the full-orbit finite zeta `Z_P(k)`, and `F_{p^2}` arithmetic — Frobenius, trace, norm, the unit circle `U_{p+1}`, the square roots of `-1`).
- `validate.py` — checks every claim against its proposition/theorem number; prints `PASS`/`FAIL` and a summary.
- `figures.py` — regenerates `fig_obstruction.pdf`, `fig_emergence.pdf`, `fig_rh_correspondence.pdf`.

## Run
```
pip install numpy sympy matplotlib
python3 validate.py     # -> 36/36 checks passed
python3 figures.py      # -> three PDFs
```

## What `validate.py` checks (claim → result)
| Paper | Claim | Check |
|---|---|---|
| Thm 3.1 | `F_P^x` = `T` cosets of the order-4 packet `mu_4` | exact, `P=13,29,37,101` |
| Thm 7.2 / Cor 7.3 | `Z_P(k)=0` for `k=1..P-2`, `=-1` on the full cycle | exact |
| Lem 9.1 / Thm 9.2 | `Phi(k)=-g^k` bijects `Z_P^o → A_P^o`, intertwined by `-1` | exact set equality |
| Lem 5.1–5.2 | `|U_{p+1}|=p+1`, Frobenius = inversion, `j` trace-0 | exact in `F_{p^2}` |
| Thm 5.5 | placement dichotomy of `sqrt(-1)` (`p≡1` real off-circle / `p≡3` imaginary on-circle) | exact, both classes |
| Cor 5.6 | `mu_4(F_p) ∩ U_{p+1} = mu_2` | exact |
| Prop 6.3 | every readout `2^{-1}+jθ` has `Tr=1`; `|L_{1/2}|=p` | exact |
| Prop 10.2 | units-chart ground state `c_p≡-1` (trig = Kluyver) | exact |
| Prop 10.3 | Parseval: rms chart-mode corr `~1/sqrt(p)` | exact mean-energy |
| Num. Obs. 10.4 | max chart-mode corr at floor (sqrt-cancellation, GRH-strength for chars mod `p`) | `p=1009`: 0.082 vs 0.031; 2-chart 0.090; Carrier `~10^5`: 0.0087 vs 0.0032 |
| Thm 10.7(a) | units-chart `R_L -> (phi/n)*Lambda`, support = prime powers, weight `(1-1/l)*log(l)` | Cesaro; `R(2) ~ (1/2)log2 = 0.347` |
| Thm 10.7(b,c) | `Lambda = mu*log` (exact `log(l)`); intertwiner `Lambda = (n/phi)*R` | exact; `(13/12)*R(13) ~ log13` |
| Thm 10.8 | emergence: Subject primes are the resonance antinodes | corr 0.904 (window) |
| Num. Obs. 10.9 | `L*(H) ~ O(H) << H^2` (horizon-scale resolution) | `L*/H in [1.2, 2.5]` |
| Prop 10.10 | prime state is the convolution inverse, not a chart eigenmode | eigvec corr 0.28 vs resonance 0.737 |
| Prop 11.1 | de-framing `2^{-1}/p=(p+1)/(2p) → 1/2` | 0.5385 → 0.50005 |
| Fig 3 | explicit-formula duality: both spectra peak at prime powers | continuum & finite means separate |

## Figures
- `fig_obstruction.pdf` — Figure 1 (§10.2): chart-mode blindness; resolving scale `L*(H)` at the horizon.
- `fig_emergence.pdf` — Figure 2 (§10.3): units-chart standing waves, flat ground state, resonance peaking at primes.
- `fig_rh_correspondence.pdf` — Figure 3 (§11): zeros on the critical line and the explicit-formula reconstruction, continuum beside finite.
