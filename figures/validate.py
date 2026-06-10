"""
validate.py
===========
Numerical validation of every computational claim in

    "Finite Zeta, Phase Amplitudes, and the Emergence of Prime Distribution
     in the Finite Ring Continuum"

Each check is labelled with the result it supports. Run:

    python3 validate.py

Prints a PASS/FAIL line per claim and a final summary. All checks are exact
finite computations except the three explicitly-statistical ones
(equidistribution obstruction, resonance correlation, resolution law), which
report the measured numbers used in the paper.
"""
import numpy as np
from math import gcd, log, isqrt
from sympy import primitive_root, isprime, factorint
import frc_core as fc

PASS = []
def check(name, ok, detail=""):
    PASS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

def corr(u, v):
    u = u - u.mean(); v = v - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

print("="*78)
print("VALIDATION  --  Finite Ring Continuum finite-zeta paper")
print("="*78)

# shared sieve / resonance tables
Qmax, Nmax = 200, 200
mu, phi = fc.sieve_mu_phi(Qmax)
cum = fc.resonance_cumulative(Qmax, Nmax, mu, phi)   # cum[L,n] = R_L(n) (units-chart resonance)

# ----------------------------------------------------------------------
print("\n-- Section 3 : Carrier coherence decomposition (Thm 3.1) --")
for P in [13, 29, 37, 101]:
    g = primitive_root(P); T = (P - 1) // 4
    iT = pow(g, T, P)
    Q4 = {pow(iT, a, P) for a in range(4)}
    cosets = {tuple(sorted({(pow(g, r, P) * x) % P for x in Q4})) for r in range(T)}
    union = set().union(*cosets)
    ok = (len(Q4) == 4 and pow(iT, 2, P) == P - 1 and len(cosets) == T
          and union == set(range(1, P)))
    check(f"F_{P}^x = T={T} cosets of mu_4 (order-4 packet)", ok,
          f"|Q4|={len(Q4)}, cosets={len(cosets)}, cover={len(union)}={P-1}")

# ----------------------------------------------------------------------
print("\n-- Section 7 : finite zeta zero-slot theorem (Thm 7.2, Cor 7.3) --")
for P in [13, 17, 29, 41]:
    zeros_ok = all(fc.finite_zeta(P, k) == 0 for k in range(1, P - 1))
    full = fc.finite_zeta(P, 0)                      # k = 0 (== P-1 slot)
    check(f"Z_{P}(k)=0 for k=1..{P-2}, =-1 on full cycle", zeros_ok and full == P - 1,
          f"nontrivial slots vanish; Z_{P}(0)={full}=P-1")

# ----------------------------------------------------------------------
print("\n-- Section 9 : structural complementarity (Lem 9.1, Thm 9.2) --")
for P in [13, 29, 53]:
    g = primitive_root(P)
    # log_g(-1) = (P-1)/2
    log_m1 = next(k for k in range(P - 1) if pow(g, k, P) == P - 1)
    # Phi(k) = -g^k  maps the P-2 zero-slots onto A_P^o = F_P^x \ {-1} = {1..P-2}
    image = {(-pow(g, k, P)) % P for k in range(1, P - 1)}
    target = set(range(1, P)) - {P - 1}
    ok = (log_m1 == (P - 1) // 2 and len(image) == P - 2 and image == target)
    check(f"Phi(k)=-g^k bijects Z_{P}^o -> A_{P}^o, intertwiner -1", ok,
          f"log_g(-1)={log_m1}=(P-1)/2, |image|={len(image)}=P-2, onto A^o")

# ----------------------------------------------------------------------
print("\n-- Section 5 : quadratic extension F_{p^2}, placement (Lem 5.1-5.2, Thm 5.5, Cor 5.6) --")
for p, lab in [(13, "p=1 mod 4 (symmetry-complete)"), (11, "p=3 mod 4 (contrast)")]:
    d = fc.nonresidue(p)
    j = (0, 1)
    # Lemma 5.2: j^2 = d, Tr(j)=0, conj(j) = -j
    lem52 = (fc.fp2_mul(j, j, d, p) == (d, 0) and fc.fp2_trace(j, p) == 0
             and fc.fp2_conj(j, p) == (0, (-1) % p))
    # Lemma 5.1: |U_{p+1}| = p+1 and Frobenius = inversion there
    U = fc.unit_circle(p, d)
    inv_ok = all(fc.fp2_conj(z, p) == fc.fp2_pow(z, fc.fp2_order(z, d, p) - 1, d, p) for z in U[:20])
    lem51 = (len(U) == p + 1 and inv_ok)
    # Thm 5.5 placement of sqrt(-1)
    i = fc.sqrt_minus1(p, d)
    Ni = fc.fp2_norm(i, d, p)
    real = (i[1] == 0)
    has4 = any(fc.fp2_order(z, d, p) == 4 for z in U)
    if p % 4 == 1:
        placement = (real and Ni == p - 1 and (p + 1) % 4 == 2 and not has4)
        pdetail = f"sqrt(-1) real, N={Ni}=-1 (off circle), U_(p+1) no order-4"
    else:
        placement = (not real and Ni == 1 and (p + 1) % 4 == 0 and has4)
        pdetail = f"sqrt(-1) imaginary, N={Ni}=1 (on circle), 4|(p+1)"
    # Cor 5.6 (p=1 mod 4): mu_4(F_p) meets U_{p+1} only in mu_2
    if p % 4 == 1:
        mu4 = [(1, 0), i, ((-1) % p, 0), fc.fp2_conj(i, p) if i[1] else ((-i[0]) % p, 0)]
        on_circle = [z for z in mu4 if fc.fp2_norm(z, d, p) == 1]
        cor56 = (sorted(on_circle) == sorted([(1, 0), ((-1) % p, 0)]))
    else:
        cor56 = True
    check(f"{lab}: Lemmas 5.1-5.2", lem51 and lem52,
          f"|U_(p+1)|={len(U)}=p+1, Frobenius=inversion, j trace-0")
    check(f"{lab}: placement Thm 5.5", placement, pdetail)
    if p % 4 == 1:
        check(f"{lab}: Cor 5.6  mu_4 ∩ U_(p+1) = mu_2", cor56, "{1,-1} only")

# ----------------------------------------------------------------------
print("\n-- Section 6 : finite critical line incidence (Prop 6.3) --")
for p in [13, 53, 101]:
    half = pow(2, p - 2, p)                          # 2^{-1} mod p
    line = [(half, b) for b in range(p)]             # L_{1/2}
    tr_ok = all(fc.fp2_trace(z, p) == 1 for z in line)
    check(f"p={p}: every readout 2^-1 + j theta has Tr=1; |L_1/2|=p", tr_ok and len(line) == p,
          f"2^-1={half}, Tr=1 identically, {len(line)} points")

# ----------------------------------------------------------------------
print("\n-- Section 10.1 : latitude additive over factorization --")
for p in [1009]:
    P2 = 1213
    lp = fc.dlog_table(primitive_root(p), p)
    lP = fc.dlog_table(primitive_root(P2), P2)
    H = isqrt(min(p, P2))
    def add_ok(lam, mod):
        return all((lam[a] + lam[b]) % (mod - 1) == lam[(a * b) % mod] % (mod - 1)
                   for a in range(2, H + 1) for b in range(2, mod // a + 1) if b >= 2 and a * b < mod)
    check(f"lambda_p additive over window products (p={p})", add_ok(lp, p))
    check(f"lambda_P additive over window products (P={P2})", add_ok(lP, P2))

# ----------------------------------------------------------------------
print("\n-- Section 10.1 : units-chart ground state (Prop 10.2) --")
for p in [13, 53, 101]:
    flat = all(fc.ramanujan(p, n, mu if n <= Qmax else fc.sieve_mu_phi(p)[0]) == -1
               for n in range(1, p))
    # also confirm the trigonometric form is real and equals the Kluyver value
    mup, _ = fc.sieve_mu_phi(p)
    real_ok = all(abs(fc.ramanujan_trig(p, n) - fc.ramanujan(p, n, mup)) < 1e-9 for n in range(1, p))
    val0 = fc.ramanujan(p, 0, mup) if p <= Qmax else None
    check(f"c_p(n) = -1 for n!=0 (flat ground state), real (p={p})", flat and real_ok,
          "trig form = Kluyver integer = -1 off origin")

# ----------------------------------------------------------------------
print("\n-- Section 10.2 : equidistribution obstruction (Prop 10.3 / Num. Obs. 10.4) --")
p = 1009; P2 = 1213
lp = fc.dlog_table(primitive_root(p), p)
lP = fc.dlog_table(primitive_root(P2), P2)
N = p - 1
ang_p = np.array([lp[n] / (p - 1) for n in range(2, N)])
pind = np.array([1.0 if isprime(n) else 0.0 for n in range(2, N)])
floor = 1 / np.sqrt(N)
best1 = max(abs(corr(pind, np.cos(2 * np.pi * j * ang_p))) for j in range(1, (p - 1) // 2))
ang_P = np.array([lP[n] / (P2 - 1) for n in range(2, N)])
best2 = max(abs(corr(pind, np.cos(2 * np.pi * (j * ang_p + k * ang_P))))
            for j in range(0, 25) for k in range(0, 25) if (j, k) != (0, 0))
rn = range(2, Nmax + 1)                              # resonance lives in the no-wrap window
res_corr = corr(np.array([1.0 if isprime(n) else 0.0 for n in rn]),
                np.array([cum[80, n] for n in rn]))
check("single chart blind: max|corr| ~ noise floor", best1 < 10 * floor,
      f"max chart-mode corr={best1:.3f}, floor={floor:.3f}")
check("joint two-chart blind", best2 < 0.25, f"max 2-chart corr={best2:.3f}")
check("resonance resolves primes in its window (contrast)", res_corr > 0.8,
      f"corr(prime, R_80) over n<=200 = {res_corr:.3f}")

# ----------------------------------------------------------------------
print("\n-- Section 10.3 : the resonance and the von Mangoldt weight (Thm 10.7) --")
# dedicated higher bandwidth for the conditional-convergence limits
Qhi = 4000; muhi, phihi = fc.sieve_mu_phi(Qhi); cumhi = fc.resonance_cumulative(Qhi, 14, muhi, phihi)
pps = [2, 3, 5, 7, 11, 13]
rhi = {n: float(cumhi[1:Qhi + 1, n].mean()) for n in pps}   # Cesaro mean of R_L (conditional convergence)
# (a) units-chart form:  R_L -> (phi(n)/n) Lambda(n),  NOT Lambda(n)
favors_phi = all(abs(rhi[n] - phihi[n] / n * fc.von_mangoldt(n)) < abs(rhi[n] - fc.von_mangoldt(n)) for n in pps)
check("units-chart form  R_L -> (phi/n)*Lambda  (not Lambda)", favors_phi,
      f"R(2)={rhi[2]:.3f} near phi/2*log2={0.5 * fc.von_mangoldt(2):.3f}, not log2={fc.von_mangoldt(2):.3f}")
# (b) convolution-inverse form:  mu*log = Lambda  (exact)
conv_ok = all(abs(fc.mobius_log(n, mu) - fc.von_mangoldt(n)) < 1e-9 for n in range(2, Nmax + 1))
check("convolution-inverse form  Lambda = mu*log  (exact)", conv_ok,
      "Lambda(n) = -sum_{d|n} mu(d) log d = von Mangoldt for all n")
# (c) intertwiner:  (n/phi(n)) R -> Lambda = log ell
renorm = {n: n / phihi[n] * rhi[n] for n in pps}
renorm_ok = all(abs(renorm[n] - fc.von_mangoldt(n)) < 0.02 for n in pps)
check("intertwiner  (n/phi)*R -> Lambda = log ell", renorm_ok,
      f"(2/1)*R(2)={renorm[2]:.4f} vs log2={fc.von_mangoldt(2):.4f}; (13/12)*R(13)={renorm[13]:.4f} vs log13={fc.von_mangoldt(13):.4f}")

# ----------------------------------------------------------------------
print("\n-- Section 10.3 : horizon-scale resolution (Num. Obs. 10.9) --")
def Lstar(H):
    pp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) > 0]
    npp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) == 0]
    for L in range(2, Qmax + 1):
        if min(cum[L, n] for n in pp) > max(abs(cum[L, n]) for n in npp):
            return L
    return None
for H in [12, 20, 31]:
    Ls = Lstar(H)
    check(f"L*({H}) ~ O(H), << field scale p~H^2", Ls is not None and Ls < 6 * H,
          f"L*={Ls}, L*/H={Ls/H:.2f}, vs H^2={H*H}")

# ----------------------------------------------------------------------
print("\n-- Section 10.3 : prime state is not a chart spectral mode (Prop 10.10) --")
H = 60; L = 120
mu2, phi2 = fc.sieve_mu_phi(L)
C = fc.cq_table(L, H, mu2)[1:L + 1, 1:H + 1]
w = np.array([mu2[q] / phi2[q] if phi2[q] else 0.0 for q in range(1, L + 1)])
G = C.T @ np.diag(w) @ C; G = (G + G.T) / 2
vals, vecs = np.linalg.eigh(G)
order = np.argsort(-np.abs(vals))
pind_w = np.array([1.0 if isprime(n) else 0.0 for n in range(1, H + 1)])
eig_corr = max(abs(corr(vecs[:, order[r]], pind_w)) for r in range(4))
res_vec = np.array([cum[min(L, Qmax), n] for n in range(1, H + 1)])
res_corr2 = corr(res_vec, pind_w)
check("operator eigenvectors do NOT carry the prime mode", eig_corr < 0.35,
      f"max leading-eigvec corr={eig_corr:.3f}")
check("Mobius-weighted superposition DOES (convolution inverse)", res_corr2 > 0.6,
      f"corr(R_L, primes)={res_corr2:.3f}")

# ----------------------------------------------------------------------
print("\n-- Section 11 : de-framing of the critical real part (Prop 11.1) --")
vals_df = [(pp, (pp + 1) / (2 * pp)) for pp in [13, 53, 101, 409, 1009, 10007]]
monotone = all(vals_df[i][1] > vals_df[i + 1][1] for i in range(len(vals_df) - 1))
limit_ok = abs(vals_df[-1][1] - 0.5) < 1e-3 and all(v > 0.5 for _, v in vals_df)
check("2^-1/p = (p+1)/(2p) -> 1/2  (de-framing limit)", monotone and limit_ok,
      "  ".join(f"p={pp}:{v:.5f}" for pp, v in vals_df))

# ----------------------------------------------------------------------
print("\n-- Section 11 : explicit-formula duality (Fig 3) --")
gammas = np.array([14.134725,21.022040,25.010858,30.424876,32.935062,37.586178,40.918719,
43.327073,48.005151,49.773832,52.970321,56.446248,59.347044,60.831778,65.112544,67.079811,
69.546402,72.067158,75.704691,77.144840,79.337375,82.910381,84.735493,87.425275,88.809111,
92.491899,94.651344,95.870634,98.831194,101.317851])
def Rcont(x): return -sum(np.cos(g * np.log(x)) for g in gammas)
pp = [n for n in range(2, 31) if fc.von_mangoldt(n) > 0]
npp = [n for n in range(2, 31) if fc.von_mangoldt(n) == 0]
cont_ok = np.mean([Rcont(n) for n in pp]) > np.mean([Rcont(n) for n in npp])
fin_ok = np.mean([cum[120, n] for n in pp]) > np.mean([cum[120, n] for n in npp])
check("continuum zero-spectrum peaks at prime powers", cont_ok,
      f"mean@pp={np.mean([Rcont(n) for n in pp]):.2f} > mean@comp={np.mean([Rcont(n) for n in npp]):.2f}")
check("finite resonance peaks at prime powers", fin_ok,
      f"mean@pp={np.mean([cum[120,n] for n in pp]):.2f} > mean@comp={np.mean([cum[120,n] for n in npp]):.2f}")

# ----------------------------------------------------------------------
print("\n" + "="*78)
print(f"SUMMARY:  {sum(PASS)}/{len(PASS)} checks passed.")
print("="*78)
