"""
figures.py
==========
Regenerates the three figures of the finite-zeta paper from frc_core:

    fig_obstruction.pdf        (Figure 1, Section 10.2)
    fig_emergence.pdf          (Figure 2, Section 10.3)
    fig_rh_correspondence.pdf  (Figure 3, Section 11)

Run:  python3 figures.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gcd
from sympy import primitive_root, isprime
import frc_core as fc

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
OUT = "/home/claude/frc"
CONT, FIN = "#b0341f", "#3b34a8"

# shared resonance machinery (covers all three figures)
Qmax, Nmax = 320, 60
mu, phi = fc.sieve_mu_phi(Qmax)
cum = fc.resonance_cumulative(Qmax, Nmax, mu, phi)        # cum[L,n] = Lambda_L(n)

def corr(u, v):
    u = u - u.mean(); v = v - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


# =====================================================================
# FIGURE 1 : equidistribution obstruction + resolving scale
# =====================================================================
def figure_obstruction():
    p = 1009
    lp = fc.dlog_table(primitive_root(p), p); N = p - 1
    ang = np.array([lp[n] / (p - 1) for n in range(2, N)])
    pind = np.array([1.0 if isprime(n) else 0.0 for n in range(2, N)])
    cs = [abs(corr(pind, np.cos(2 * np.pi * j * ang))) for j in range(1, (p - 1) // 2)]
    floor = 1 / np.sqrt(N)

    Hs = list(range(6, 51, 2)); Ls = []
    for H in Hs:
        pp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) > 0]
        npp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) == 0]
        Lstar = np.nan
        for L in range(2, Qmax + 1):
            if min(cum[L, n] for n in pp) > max(abs(cum[L, n]) for n in npp):
                Lstar = L; break
        Ls.append(Lstar)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].plot(range(1, (p - 1) // 2), cs, color="#1f9e8a", lw=0.7)
    ax[0].axhline(floor, color="#888", ls="--", lw=1, label=r"noise floor $1/\sqrt{N}$")
    ax[0].axhline(0.90, color=FIN, ls="-", lw=1.8, label=r"resonance ($0.90$)")
    ax[0].set_xlabel(r"chart mode index $j$"); ax[0].set_ylabel("|corr with primes|"); ax[0].set_ylim(0, 1)
    ax[0].set_title("Chart modes are blind to primality", fontsize=9.5)
    ax[0].legend(fontsize=8, loc="center right", frameon=False)

    ax[1].plot(Hs, Ls, "o-", color=FIN, lw=1.4, ms=4, label=r"resolving scale $L^*(H)$")
    ax[1].plot(Hs, Hs, "--", color="#1f9e8a", lw=1.2, label=r"horizon $H=\lfloor\sqrt{p}\,\rfloor$")
    ax[1].plot(Hs, [h * h for h in Hs], ":", color="#b00", lw=1.5, label=r"field $p\sim H^2$")
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"window horizon $H$"); ax[1].set_ylabel("frequency scale")
    ax[1].set_title("Resolving band sits at the horizon", fontsize=9.5)
    ax[1].legend(fontsize=8, loc="upper left", frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_obstruction.pdf", bbox_inches="tight"); plt.close()
    print("fig_obstruction.pdf")


# =====================================================================
# FIGURE 2 : units-chart standing waves + emergent resonance
# =====================================================================
def figure_emergence():
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.0), gridspec_kw={"height_ratios": [1, 1.2]})
    xx = np.linspace(2, 40, 1500)
    for q, c in [(6, "#5b53c9"), (8, "#1f9e8a"), (12, "#cf6a1f")]:
        ax[0].plot(xx, [fc.ramanujan_trig(q, x) for x in xx], color=c, lw=1.2, label=r"$c_{%d}(n)$" % q)
    ax[0].axhline(-1, color="#b00", lw=1.5, ls="--", label=r"$c_p\equiv-1$ (ground state)")
    ax[0].set_xlim(2, 40); ax[0].set_ylim(-5, 5); ax[0].set_ylabel("amplitude")
    ax[0].legend(fontsize=8, ncol=4, loc="lower center", frameon=False)
    ax[0].set_title(r"Units-chart standing waves $c_q(n)=\sum_{a\in(\mathbb{Z}/q)^\times}e^{2\pi i an/q}$ on the additive meridian")

    L = 80; ns = np.arange(2, 41)
    for n in ns:
        if isprime(int(n)):
            ax[1].axvline(n, color="#5b53c9", alpha=0.16, lw=6, zorder=0)
    ax[1].plot(ns, [cum[L, int(n)] for n in ns], color="#222", lw=1.2, marker="o", ms=3, zorder=3)
    pr = [n for n in ns if isprime(int(n))]
    ax[1].scatter(pr, [cum[L, int(n)] for n in pr], color="#5b53c9", s=40, zorder=5,
                  label="Subject-realized primes (antinodes)")
    ax[1].axhline(0, color="#999", lw=0.7); ax[1].set_xlim(2, 40)
    ax[1].set_xlabel(r"position $n$ on the additive meridian (vicinity of origin)")
    ax[1].set_ylabel("resonance"); ax[1].legend(fontsize=8.5, loc="upper right", frameon=True)
    ax[1].set_title(r"units-chart resonance $R_L(n)=\sum_{q\leq L}\frac{\mu(q)}{\varphi(q)}c_q(n)$, antinodes at primes")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_emergence.pdf", bbox_inches="tight"); plt.close()
    print("fig_emergence.pdf")


# =====================================================================
# FIGURE 3 : Riemann picture vs finite analogue
# =====================================================================
GAMMAS = np.array([14.134725,21.022040,25.010858,30.424876,32.935062,37.586178,40.918719,
43.327073,48.005151,49.773832,52.970321,56.446248,59.347044,60.831778,65.112544,67.079811,
69.546402,72.067158,75.704691,77.144840,79.337375,82.910381,84.735493,87.425275,88.809111,
92.491899,94.651344,95.870634,98.831194,101.317851])

def figure_rh():
    p = 101; half = (p + 1) // 2; xmax = 30
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))

    # (a) continuum zeros on the critical line
    A = ax[0, 0]; A.axvspan(0, 1, color="#eee", zorder=0); A.axvline(0.5, color=CONT, lw=1.8)
    A.scatter(np.full_like(GAMMAS, 0.5), GAMMAS, color=CONT, s=26, zorder=4)
    A.set_xlim(-0.6, 1.6); A.set_ylim(0, 103); A.set_xlabel(r"$\mathrm{Re}\,s$"); A.set_ylabel(r"$\mathrm{Im}\,s=\gamma$")
    A.set_title(r"Continuum: nontrivial zeros of $\zeta$ on $\mathrm{Re}\,s=\frac{1}{2}$", color=CONT)
    A.text(0.52, 96, r"$\rho=\frac{1}{2}+i\gamma$", color=CONT, fontsize=9)

    # (b) finite framed zeros on the finite critical line
    B = ax[0, 1]; B.axvspan(0, 1, color="#eee", zorder=0); xc = half / p; B.axvline(xc, color=FIN, lw=1.8)
    ks = np.arange(1, p - 1)
    B.scatter(np.full_like(ks, xc, dtype=float), ks, color=FIN, s=22, zorder=4)
    B.set_xlim(-0.6, 1.6); B.set_ylim(0, 103); B.set_xlabel(r"normalized real part $a/p$")
    B.set_ylabel(r"transverse slot $\theta(k)$")
    B.set_title(r"Finite ($p=%d$): all framed zeros on $\mathrm{Re}=2^{-1}$" % p, color=FIN)
    B.text(xc + 0.03, 96, r"$\rho(k)=2^{-1}+j\,\theta(k)$", color=FIN, fontsize=9)
    B.text(xc + 0.03, 6, r"$\frac{2^{-1}}{p}=\frac{p+1}{2p}=\frac{1}{2}+\frac{1}{2p}$", color=FIN, fontsize=8.5)

    prime_powers = [n for n in range(2, xmax + 1) if fc.von_mangoldt(n) > 0]

    # (c) continuum: the zeros reconstruct the primes
    C2 = ax[1, 0]
    def Rcont(x): return -sum(np.cos(g * np.log(x)) for g in GAMMAS)
    xs = np.linspace(2.0, xmax, 2600)
    for n in prime_powers: C2.axvline(n, color="#bbb", lw=0.8, zorder=0)
    C2.plot(xs, [Rcont(x) for x in xs], color=CONT, lw=1.1)
    C2.set_xlim(2, xmax); C2.set_xlabel(r"$x$"); C2.set_ylabel(r"$-\sum_{\gamma}\cos(\gamma\log x)$")
    C2.set_title(r"Continuum: the zero spectrum reconstructs the primes", color=CONT)
    C2.text(0.98, 0.93, "peaks at prime powers", transform=C2.transAxes, ha="right", fontsize=8, color="#555")

    # (d) finite: the units-chart spectrum reconstructs the primes
    D = ax[1, 1]; ns = list(range(2, xmax + 1)); L = 120
    for n in prime_powers: D.axvline(n, color="#bbb", lw=0.8, zorder=0)
    D.stem(ns, [cum[L, n] for n in ns], linefmt=FIN, markerfmt="o", basefmt=" ")
    D.set_xlim(2, xmax); D.set_xlabel(r"$n$")
    D.set_ylabel(r"$R_L(n)=\sum_{q\leq L}\frac{\mu(q)}{\varphi(q)}c_q(n)$")
    D.set_title(r"Finite: the units-chart spectrum reconstructs the primes", color=FIN)
    D.text(0.98, 0.93, r"peaks at prime powers $\to\log\ell$", transform=D.transAxes, ha="right", fontsize=8, color="#555")

    plt.tight_layout(); plt.savefig(f"{OUT}/fig_rh_correspondence.pdf", bbox_inches="tight"); plt.close()
    print("fig_rh_correspondence.pdf")


if __name__ == "__main__":
    figure_obstruction()
    figure_emergence()
    figure_rh()
    print("all figures regenerated.")
