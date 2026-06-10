#!/usr/bin/env python3
"""
Reproduces fig_operator_spectrum.pdf (Figure 5 / 6 of the manuscript):
"The scale-evolution spectrum as a discrete spectrum, with zeta never evaluated."

Three panels, all built from the von Mangoldt comb (zeta is never evaluated):

  LEFT   Secular condition. The trace counting function
             N(T) = theta(T)/pi + 1 + S_comb(T)
         (archimedean scale-phase + prime comb) steps through n-1/2 exactly at
         the Riemann zeros.

  CENTER Eigenvalues of the explicit matrix. The companion (colleague) matrix C
         of the comb-built secular determinant  Xi(T) = cos(pi N(T))  has the
         Riemann heights as its eigenvalues; mean |lambda - gamma| = 5.9e-5 at
         dimension 520 (4.5e-5 at 620), converging to the comb-truncation floor (inset).

  RIGHT  Additive operator is gauge-trivial. The dilation generator plus the comb
         as a real potential,  -i d/du + V_comb,  has a uniform spectrum
         (the potential is gauged into a boundary phase), unlike the irregular
         Riemann zeros -- so the comb must enter the trace, not a local potential.

Dependencies: numpy, scipy, matplotlib.  Runtime ~1-3 min (dominated by the
companion-matrix degree scan for the convergence inset).
"""

import math
import numpy as np
from scipy.special import loggamma
from numpy.polynomial import chebyshev as Ch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# First ten nontrivial zero heights (reference values for comparison only;
# they are NOT used in any construction below).
# ----------------------------------------------------------------------
TRUE_G = np.array([14.134725142, 21.022039639, 25.010857580, 30.424876126,
                   32.935061588, 37.586178159, 40.918719012, 43.327073281,
                   48.005150881, 49.773832478])

# ----------------------------------------------------------------------
# The von Mangoldt comb  (the only arithmetic input; zeta never evaluated)
#   S_comb(T) = -(1/pi) * sum_{n<=N} Lambda(n)/(sqrt(n) log n) * sin(T log n)
# with a raised-cosine taper in log n to suppress truncation ringing.
# ----------------------------------------------------------------------
def build_comb(N):
    comp = np.zeros(N + 1, bool); comp[:2] = True
    for i in range(2, int(N**0.5) + 1):
        if not comp[i]:
            comp[i * i::i] = True
    primes = np.nonzero(~comp)[0]
    Lam = np.zeros(N + 1)
    for p in primes:
        lp = math.log(p); pk = int(p)
        while pk <= N:
            Lam[pk] = lp; pk *= p
    nz = np.nonzero(Lam)[0]
    logn = np.log(nz)
    taper = 0.5 * (1.0 + np.cos(np.pi * logn / math.log(N)))
    amp = (Lam[nz] / (np.sqrt(nz) * logn)) * taper
    return amp, logn

N_COMB = 1_000_000
AMP, LOGN = build_comb(N_COMB)

def theta(T):
    """Riemann-Siegel theta: phase of the archimedean (Gamma) factor; NOT zeta."""
    T = np.asarray(T, float)
    return np.imag(loggamma(0.25 + 0.5j * T)) - 0.5 * T * math.log(math.pi)

def Ncount(T):
    """Trace counting function  N(T) = theta(T)/pi + 1 + S_comb(T)."""
    T = np.atleast_1d(np.asarray(T, float))
    S = np.empty(len(T))
    for s in range(0, len(T), 128):                       # chunk the outer product
        block = T[s:s + 128]
        S[s:s + 128] = -(1.0 / math.pi) * (AMP[:, None] * np.sin(np.outer(LOGN, block))).sum(0)
    return theta(T) / math.pi + 1.0 + S

# ----------------------------------------------------------------------
# PANEL A data: the trace staircase on [8, 52]
# ----------------------------------------------------------------------
Tg = np.arange(8.0, 52.0, 0.01)
NB = theta(Tg) / math.pi + 1.0          # archimedean scale-phase alone (smooth)
NC = Ncount(Tg)                         # archimedean phase + comb (the staircase)

# ----------------------------------------------------------------------
# PANEL B data: companion (colleague) matrix of  Xi(T) = cos(pi N(T))  on [10, 52]
# Its eigenvalues are the Riemann heights.
# ----------------------------------------------------------------------
TA, TB = 10.0, 52.0

def colleague(deg, return_matrix=False):
    # Chebyshev fit of Xi on [TA, TB] at Chebyshev nodes
    xs = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))    # nodes in [-1, 1]
    T = 0.5 * (TB + TA) + 0.5 * (TB - TA) * xs
    c = Ch.chebfit(xs, np.cos(math.pi * Ncount(T)), deg)
    # build the colleague matrix explicitly (eigenvalues = roots of the Cheb series)
    n = len(c) - 1
    a = c[:-1] / c[-1]
    M = np.zeros((n, n))
    M[0, 1] = 1.0
    for k in range(1, n - 1):
        M[k, k - 1] = 0.5
        M[k, k + 1] = 0.5
    M[n - 1, n - 2] = 0.5
    M[-1, :] -= 0.5 * a
    ev = np.linalg.eigvals(M)
    ev = ev[np.abs(ev.imag) < 1e-5].real
    ev = ev[(ev > -1) & (ev < 1)]                                  # keep roots in window
    roots = np.sort(0.5 * (TB + TA) + 0.5 * (TB - TA) * ev)
    return (roots, M) if return_matrix else roots

def mean_err(roots):
    rec = np.array([roots[np.argmin(np.abs(roots - g))] for g in TRUE_G])
    return float(np.mean(np.abs(rec - TRUE_G)))

eig520 = colleague(520)
print(f"companion matrix (deg 520): {len(eig520)} eigenvalues in window, "
      f"mean|lambda-gamma| = {mean_err(eig520):.2e}")

conv_degrees = [180, 260, 340, 420, 520, 620]
convD = np.array([(d, mean_err(colleague(d))) for d in conv_degrees])

# ----------------------------------------------------------------------
# PANEL C data: additive operator  H = -i d/du + V_comb  on a periodic log-grid.
# Gauge argument: for any real V, H is unitarily equivalent to -i d/du with a
# shifted boundary phase, so the spectrum stays uniform.  Shown numerically.
# ----------------------------------------------------------------------
def additive_spectrum(U=15.0, M=2048, potential_scale=5.0):
    du = U / M
    k = np.fft.fftfreq(M, d=du) * 2 * np.pi
    F = np.fft.fft(np.eye(M), axis=0) / np.sqrt(M)
    D = F.conj().T @ np.diag(k) @ F                  # -i d/du in position basis (Hermitian)
    D = 0.5 * (D + D.conj().T)
    # comb potential: weight Lambda(n)/sqrt(n) placed at u = log n, for log n <= U
    Np = int(math.e**U) + 1
    comp = np.zeros(Np + 1, bool); comp[:2] = True
    for i in range(2, int(Np**0.5) + 1):
        if not comp[i]:
            comp[i * i::i] = True
    primes = np.nonzero(~comp)[0]
    V = np.zeros(M)
    for p in primes:
        lp = math.log(p); pk = int(p)
        while pk <= Np:
            j = int(round(math.log(pk) / du)) % M
            V[j] += lp / math.sqrt(pk) / du
            pk *= p
    H = D + np.diag(potential_scale * V)
    H = 0.5 * (H + H.conj().T)
    return np.sort(np.linalg.eigvalsh(H))

ev_add = additive_spectrum()

# ----------------------------------------------------------------------
# The figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# --- Panel A: trace staircase ---
m = (Tg >= 11) & (Tg <= 40)
ax[0].plot(Tg[m], NC[m], color="#10325f", lw=1.4,
           label=r"$\frac{\theta(T)}{\pi}+1+S_{\mathrm{comb}}$")
ax[0].plot(Tg[m], NB[m], color="#9ec5e8", lw=1.3,
           label=r"$\frac{\theta(T)}{\pi}+1$ (archimedean)")
for n in range(2, 9):
    ax[0].axhline(n - 0.5, color="#bbb", lw=0.6, ls=":")
for g in TRUE_G[(TRUE_G > 11) & (TRUE_G < 40)]:
    ax[0].axvline(g, color="#c0392b", lw=0.8, ls="--", alpha=0.7)
ax[0].set_xlim(11, 40); ax[0].set_ylim(1, 9)
ax[0].set_xlabel(r"$T$"); ax[0].set_ylabel("eigenvalue count")
ax[0].set_title(r"Secular condition: count $=n-\frac{1}{2}$ at zeros", fontsize=10.5)
ax[0].legend(fontsize=8, loc="upper left")

# --- Panel B: companion-matrix eigenvalues on the heights, with convergence inset ---
ax[1].vlines(TRUE_G, 0, 1, color="#c0392b", ls="--", lw=1.1, label="Riemann heights")
ax[1].plot(eig520, 0.5 * np.ones_like(eig520), "o", color="#10325f", ms=7,
           label="matrix eigenvalues")
ax[1].set_xlim(10, 52); ax[1].set_ylim(0, 1); ax[1].set_yticks([])
ax[1].set_xlabel(r"$\gamma$")
ax[1].set_title("Eigenvalues of the explicit matrix", fontsize=10.5)
ax[1].legend(fontsize=8, loc="upper center")
ax[1].text(0.5, 0.12,
           "$520\\times520$ matrix from the comb\n"
           r"mean $|\lambda-\gamma|=5.9\times10^{-5}$",
           transform=ax[1].transAxes, ha="center", fontsize=8.5, color="#333")
axin = ax[1].inset_axes([0.12, 0.62, 0.42, 0.32])
axin.semilogy(convD[:, 0], convD[:, 1], "o-", color="#1f6b4a", lw=1.1, ms=3.5)
axin.axhline(4.4e-5, color="#c0392b", lw=0.7, ls=":")
axin.set_xlabel("dim", fontsize=7); axin.set_ylabel("err", fontsize=7)
axin.tick_params(labelsize=6); axin.set_title("convergence", fontsize=7)

# --- Panel C: additive operator uniform (gauge-trivial) vs irregular zeros ---
eva = np.sort(ev_add[(ev_add > 10) & (ev_add < 50)])
ax[2].vlines(eva, 0.55, 0.95, color="#1f6b4a", lw=1.0)
ax[2].vlines(TRUE_G[(TRUE_G > 10) & (TRUE_G < 50)], 0.05, 0.45, color="#c0392b", lw=1.4)
ax[2].set_xlim(10, 50); ax[2].set_ylim(0, 1)
ax[2].set_yticks([0.25, 0.75])
ax[2].set_yticklabels(["Riemann\nzeros", "dilation\n+ comb"], fontsize=8.5)
ax[2].set_xlabel(r"$\gamma$")
ax[2].set_title("Additive operator: uniform (excluded)", fontsize=10.5)
ax[2].text(0.5, 0.985, "comb potential gauged into a boundary phase",
           transform=ax[2].transAxes, ha="center", va="top", fontsize=8, color="#444")

plt.tight_layout()
plt.savefig("fig_operator_spectrum.pdf", bbox_inches="tight")
print("wrote fig_operator_spectrum.pdf")
