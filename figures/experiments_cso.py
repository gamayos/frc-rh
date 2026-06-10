"""
experiments_cso.py
==================
Repeat the finite-zeta numerical experiments in a genuinely three-tier
Carrier / Subject / Object hierarchy with real scale separation:

    Object   O = F_13          (a small symmetry-complete shell, 13 = 4*3+1)
    Subject  p ~ 1000          (frame; horizon H_p = floor(sqrt p) contains the Object)
    Carrier  P ~ 100000        (medium; hosts the spectral zeros and the flat ground state)

Each phenomenon is tested at the tier where it lives:
  - Carrier : zero-slot, order-4 packet decomposition, slot complementarity,
              and the equidistribution obstruction (FFT over the discrete-log chart).
  - Subject : F_{p^2} placement, finite critical line, horizon, de-framing bias.
  - Object  : the Mobius resonance reconstructs the Object's primes inside the
              Subject's ordered comparison layer.
"""
import numpy as np
from math import gcd, log, isqrt
from sympy import primitive_root, isprime, nextprime, sqrt_mod, factorint
import frc_core as fc

def banner(t): print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)
def line(ok, name, detail=""):
    print(f"  [{'OK ' if ok else 'XX '}] {name}" + (f"  --  {detail}" if detail else ""))
def corr(u, v):
    u = u - u.mean(); v = v - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

# ----------------------------------------------------------------------
# Fix the three tiers
# ----------------------------------------------------------------------
O = 13
p = 1009
P = 100000
while True:
    P = int(nextprime(P))
    if P % 4 == 1:
        break

Hp, HP = isqrt(p), isqrt(P)
print(f"Object   O = F_{O}      ({O} = 4*{(O-1)//4}+1)")
print(f"Subject  p = {p}     ({p} = 4*{(p-1)//4}+1),  horizon H_p = floor(sqrt p) = {Hp}")
print(f"Carrier  P = {P}   ({P} = 4*{(P-1)//4}+1),  horizon H_P = floor(sqrt P) = {HP}")
print(f"scale separation:  P/p = {P/p:.1f},   p/O = {p/O:.1f}")

# primality sieve up to P (for the Carrier obstruction)
sieve = np.ones(P, dtype=bool); sieve[:2] = False
for i in range(2, isqrt(P) + 1):
    if sieve[i]:
        sieve[i*i::i] = False

# ======================================================================
banner("CARRIER  (P = %d)  --  flat medium and spectral zeros" % P)

# -- zero-slot theorem (sampled, exact, via vectorized modular powers) --
def power_sum(k):
    base = np.arange(1, P, dtype=np.int64)
    res = np.ones(P - 1, dtype=np.int64); b = base.copy(); kk = k
    while kk:
        if kk & 1: res = (res * b) % P
        b = (b * b) % P; kk >>= 1
    return int(res.sum() % P)
rng = np.random.default_rng(0)
ks = sorted(rng.integers(1, P - 1, 12).tolist())
zeros_ok = all(power_sum(k) == 0 for k in ks)
full = power_sum(0)
line(zeros_ok and full == P - 1, "zero-slot theorem  Z_P(k)=0 (sampled), -1 on full cycle",
     f"12 random nontrivial slots vanish; Z_P(0)={full}=P-1")

# -- order-4 packet decomposition --
g = int(primitive_root(P)); T = (P - 1) // 4
iT = pow(g, T, P)
packet_ok = (pow(iT, 2, P) == P - 1 and pow(iT, 4, P) == 1 and len({pow(iT, a, P) for a in range(4)}) == 4)
line(packet_ok, "order-4 packet  mu_4 = {1,i,-1,-i},  F_P^x = T cosets",
     f"i^2=-1, |mu_4|=4, T=(P-1)/4={T}")

# -- structural complementarity  Phi(k) = -g^k  bijects {1..P-2} -> F_P^x\{-1} --
seen = np.zeros(P, dtype=bool); val = 1
for k in range(1, P - 1):
    val = (val * g) % P
    seen[(-val) % P] = True
img_ok = (seen[1:P - 1].all() and not seen[0] and not seen[P - 1])
line(img_ok, "complementarity  Phi(k)=-g^k : Z_P^o -> A_P^o  bijection",
     f"image = F_P^x \\ {{-1}} = {{1..{P-2}}}, intertwiner -1")

# -- equidistribution obstruction at the Carrier scale (FFT over chart) --
g_arr = np.empty(P - 1, dtype=np.int64); x = 1
for ell in range(P - 1):
    g_arr[ell] = x; x = (x * g) % P              # g_arr[ell] = g^ell  (residue at chart latitude ell)
v = sieve[g_arr].astype(float)                   # prime indicator reindexed by discrete log
N = P - 1; mu_dens = v.mean(); sd = v.std()
V = np.fft.fft(v - mu_dens)
amp = np.abs(V) / (N * sd)                        # normalized chart-mode correlation amplitudes
amp[0] = 0
floor = 1 / np.sqrt(N)
max_amp = float(amp[1:N // 2].max())
line(max_amp < 12 * floor, "obstruction: prime indicator FLAT in the Carrier chart",
     f"max chart-mode corr={max_amp:.4f}  (over {N//2} modes), noise floor 1/sqrt(N)={floor:.4f}")
print(f"        pi(P)={int(v.sum())} primes among {N} residues; "
      f"mean amplitude over all modes = {float(amp[1:N//2].mean()):.4f}")

# ======================================================================
banner("SUBJECT  (p = %d)  --  the observing frame" % p)

d = fc.nonresidue(p)
# F_{p^2} placement (Thm 5.5):  sqrt(-1) real & off circle; U_{p+1} has no order-4
r = int(sqrt_mod(-1, p))                          # real square root of -1 (exists, p = 1 mod 4)
i_el = (r, 0)
Ni = fc.fp2_norm(i_el, d, p)
# generator of U_{p+1}: raise a primitive element to the (p-1) power
def fp2_primitive(p, d):
    for a in range(2, p):
        u = (a, 1)
        if fc.fp2_order(u, d, p) == p * p - 1:
            return u
    raise RuntimeError
gen = fp2_primitive(p, d)
gen_circle = fc.fp2_pow(gen, p - 1, d, p)         # lands in U_{p+1}
ord_circle = fc.fp2_order(gen_circle, d, p)
has4 = (p + 1) % 4 == 0
place_ok = (i_el[1] == 0 and Ni == p - 1 and ord_circle == p + 1 and not has4)
line(place_ok, "F_{p^2} placement  Thm 5.5  (p = 1 mod 4)",
     f"sqrt(-1) real, N={Ni}=-1 (off circle); |U_(p+1)|={ord_circle}=p+1; 4 nmid (p+1) -> no order-4")

# finite critical line incidence (Prop 6.3)
half = pow(2, p - 2, p)
tr_ok = all(fc.fp2_trace((half, b), p) == 1 for b in range(p))
line(tr_ok, "finite critical line  L_1/2  incidence",
     f"2^-1 = {half} = (p+1)/2; readout 2^-1 + j*theta has Tr=1; |L_1/2| = p = {p}")

# Subject-realized primes and the de-framing bias
subj_primes = [n for n in range(2, Hp + 1) if isprime(n)]
print(f"        Subject-realized primes (<= H_p = {Hp}): {subj_primes}")
print(f"        de-framing bias  2^-1/p = (p+1)/(2p) = {(p+1)/(2*p):.5f}  ->  1/2")

# ======================================================================
banner("OBJECT  (O = F_%d)  --  observed inside the Subject's window" % O)

obj_primes = [n for n in range(2, O + 1) if isprime(n)]
nested = set(obj_primes).issubset(set(subj_primes))
line(nested, "nesting: Object primes are fully resolved by the Subject",
     f"Object primes {obj_primes}  subset of  Subject-realized primes (<= {Hp})")

# Object-tier zero-slot / complementarity (the small shell itself)
mu_o, _ = fc.sieve_mu_phi(O)
zslot_o = all(fc.finite_zeta(O, k) == 0 for k in range(1, O - 1)) and fc.finite_zeta(O, 0) == O - 1
go = int(primitive_root(O))
img_o = {(-pow(go, k, O)) % O for k in range(1, O - 1)} == set(range(1, O - 1))
line(zslot_o and img_o, "Object shell F_13 satisfies zero-slot + complementarity", "same theorems, smallest tier")

# the resonance reconstructs the Object's prime distribution (Subject comparison layer)
Qmax, Nmax = 60, O
mu, phi = fc.sieve_mu_phi(Qmax)
cum = fc.resonance_cumulative(Qmax, Nmax, mu, phi)
def Lstar(H):
    pp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) > 0]
    npp = [n for n in range(2, H + 1) if fc.von_mangoldt(n) == 0]
    for L in range(2, Qmax + 1):
        if min(cum[L, n] for n in pp) > max(abs(cum[L, n]) for n in npp):
            return L
    return None
Ls = Lstar(O)
ns = list(range(2, O + 1))
L_used = min(Qmax, 3 * Hp)
res_corr = corr(np.array([1.0 if isprime(n) else 0.0 for n in ns]),
                np.array([cum[L_used, n] for n in ns]))
line(Ls is not None and Ls < 6 * O, f"resonance resolves the Object's primes; L*({O}) ~ O(H)",
     f"L*({O})={Ls}  (<< Carrier scale P={P})")
line(res_corr > 0.6, "resonance correlates with Object primes",
     f"corr(R_{L_used}, primes<= {O}) = {res_corr:.3f}")
print("        resonance R_L(n) on the Object window (L=%d):" % L_used)
for n in ns:
    tag = "  <- prime" if isprime(n) else ""
    print(f"          n={n:2d}: {cum[L_used, n]:+.3f}{tag}")

# ======================================================================
banner("CROSS-SCALE SUMMARY")
print(f"  Carrier chart spectrum FLAT (max corr {max_amp:.4f} ~ floor {floor:.4f})"
      f"  =>  primes invisible in the medium")
print(f"  Subject frames the line at Re = 2^-1, bias 2^-1/p = {(p+1)/(2*p):.5f} -> 1/2")
print(f"  Object primes {obj_primes} EMERGE as the resonance antinodes "
      f"(corr {res_corr:.3f}) inside the Subject layer")
print(f"  scale ladder:   Object {O}   <   Subject horizon {Hp}   <<   Carrier {P}")
