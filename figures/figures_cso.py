"""figures_cso.py -- visualize the three-tier Carrier/Subject/Object experiments."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import isqrt
from sympy import primitive_root, isprime, nextprime
import frc_core as fc

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
CAR, SUB, OBJ = "#b0341f", "#1f7a8c", "#3b34a8"

O, p = 13, 1009
P = 100000
while True:
    P = int(nextprime(P))
    if P % 4 == 1: break
Hp = isqrt(p)

# ---- Carrier obstruction spectrum (FFT over the discrete-log chart) ----
sieve = np.ones(P, dtype=bool); sieve[:2] = False
for i in range(2, isqrt(P) + 1):
    if sieve[i]: sieve[i*i::i] = False
g = int(primitive_root(P)); N = P - 1
g_arr = np.empty(N, dtype=np.int64); x = 1
for ell in range(N):
    g_arr[ell] = x; x = (x * g) % P
v = sieve[g_arr].astype(float)
V = np.fft.fft(v - v.mean()); amp = np.abs(V) / (N * v.std()); amp[0] = 0
floor = 1 / np.sqrt(N)

# ---- resonance machinery (Object scale) ----
Qmax = 80; mu, phi = fc.sieve_mu_phi(Qmax)
cum = fc.resonance_cumulative(Qmax, Qmax, mu, phi)

fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))

# (a) Carrier: prime indicator flat in the chart
A = ax[0, 0]
idx = np.arange(1, N // 2)
sub = idx[::20]                                  # downsample for the vector PDF
A.plot(sub, amp[sub], color=CAR, lw=0.5)
A.axhline(floor, color="#888", ls="--", lw=1, label=r"noise floor $1/\sqrt{N}$")
A.axhline(0.90, color=OBJ, ls="-", lw=1.6, label=r"resonance scale ($0.90$)")
A.set_ylim(0, 1); A.set_xlabel("chart mode index $j$"); A.set_ylabel("|corr with primes|")
A.set_title(r"Carrier $P=%d$: primes flat in the chart" % P, color=CAR)
A.legend(fontsize=8, loc="center right", frameon=False)
A.text(0.96, 0.55, f"max over {N//2} modes = {amp[1:N//2].max():.4f}",
       transform=A.transAxes, ha="right", fontsize=8, color="#555")

# (b) Subject: framed zeros on the finite critical line Re = 2^-1
B = ax[0, 1]
B.axvspan(0, 1, color="#eee", zorder=0)
xc = (p + 1) / (2 * p)
B.axvline(xc, color=SUB, lw=1.8)
ks = np.arange(1, p)
B.scatter(np.full_like(ks, xc, dtype=float), ks, color=SUB, s=4, zorder=4)
B.set_xlim(-0.6, 1.6); B.set_ylim(0, p)
B.set_xlabel("normalized real part $a/p$"); B.set_ylabel(r"transverse slot $\theta(k)$")
B.set_title(r"Subject $p=%d$: framed zeros on $\mathrm{Re}=2^{-1}$" % p, color=SUB)
B.text(xc + 0.04, 0.93 * p, r"$\frac{2^{-1}}{p}=%.5f$" % xc, color=SUB, fontsize=10)
B.text(xc + 0.04, 0.84 * p, r"$\to\frac{1}{2}$", color=SUB, fontsize=10)

# (c) Object: primes emerge as resonance antinodes
C = ax[1, 0]
ns = list(range(2, O + 1)); L = 60
pr = [n for n in ns if isprime(n)]
for n in pr: C.axvline(n, color="#cdd", lw=5, alpha=0.6, zorder=0)
C.stem(ns, [cum[L, n] for n in ns], linefmt=OBJ, markerfmt="o", basefmt=" ")
C.scatter(pr, [cum[L, n] for n in pr], color=OBJ, s=45, zorder=5, label="Object primes")
C.set_xlim(1.5, O + 0.5); C.set_xlabel("$n$ on the Object window")
C.set_ylabel(r"$\Lambda_L(n)$"); C.set_title(r"Object $\mathbb{F}_{%d}$: primes are the antinodes" % O, color=OBJ)
C.legend(fontsize=8, loc="upper left", frameon=False)

# (d) scale ladder: resolving scale sits at the Subject horizon, far below the Carrier
D = ax[1, 1]
def Lstar(H):
    pp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) > 0]
    npp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) == 0]
    for Lq in range(2, Qmax + 1):
        if min(cum[Lq, n] for n in pp) > max((abs(cum[Lq, n]) for n in npp), default=-np.inf):
            return Lq
    return np.nan
Hs = list(range(4, 41, 2)); Ls = [Lstar(H) for H in Hs]
D.plot(Hs, Ls, "o-", color=OBJ, lw=1.4, ms=4, label=r"resolving scale $L^*(H)$")
D.plot(Hs, Hs, "--", color=SUB, lw=1.2, label=r"$H$")
D.axhline(P, color=CAR, lw=1.6, ls=":", label=r"Carrier $P\approx10^5$")
D.axvline(O, color=OBJ, lw=1.0, alpha=0.5); D.text(O, 1.4, " Object 13", color=OBJ, fontsize=8)
D.axvline(Hp, color=SUB, lw=1.0, alpha=0.5); D.text(Hp, 1.4, " Subject horizon 31", color=SUB, fontsize=8)
D.set_yscale("log"); D.set_xlabel("window horizon $H$"); D.set_ylabel("scale")
D.set_title("Resolving band sits at the Subject horizon", fontsize=10)
D.legend(fontsize=8, loc="center right", frameon=False)

plt.tight_layout()
plt.savefig("/home/claude/frc/fig_cso_scales.pdf", bbox_inches="tight")
plt.close()
print("fig_cso_scales.pdf written")
