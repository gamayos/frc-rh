"""
frc_core.py
===========
Core arithmetic primitives for

    "Finite Zeta, Phase Amplitudes, and the Emergence of Prime Distribution
     in the Finite Ring Continuum"  (Y. Akhtman, 2026)

Everything the validation and figure scripts need:
  - sieves for the Mobius and Euler-totient functions
  - Ramanujan / units-chart sums  c_q(n)  (Kluyver closed form and the
    trigonometric definition)
  - discrete logarithm (chart latitude) tables
  - the von Mangoldt function, its convolution-inverse form, and the
    Mobius-Ramanujan resonance  Lambda_L
  - the full-orbit finite zeta  Z_P(k)
  - finite-field arithmetic in F_{p^2} (Frobenius, trace, norm, the finite
    unit circle U_{p+1}, the square roots of -1)

No third-party dependencies beyond numpy and sympy.
"""
import numpy as np
from math import gcd, log
from sympy import primitive_root, isprime, factorint

# ----------------------------------------------------------------------
# Sieves
# ----------------------------------------------------------------------
def sieve_mu_phi(M):
    """Return (mu, phi) arrays of length M+1 (linear sieve)."""
    mu = np.ones(M + 1, dtype=int)
    phi = np.arange(M + 1)
    is_comp = np.zeros(M + 1, dtype=bool)
    primes = []
    mu[0] = 0
    for i in range(2, M + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1; phi[i] = i - 1
        for q in primes:
            if i * q > M:
                break
            is_comp[i * q] = True
            if i % q == 0:
                mu[i * q] = 0; phi[i * q] = phi[i] * q
                break
            else:
                mu[i * q] = -mu[i]; phi[i * q] = phi[i] * (q - 1)
    return mu, phi


def divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return ds


# ----------------------------------------------------------------------
# Ramanujan / units-chart sums   c_q(n) = sum_{a in (Z/q)^x} e^{2 pi i a n / q}
# ----------------------------------------------------------------------
def ramanujan(q, n, mu):
    """Kluyver closed form: c_q(n) = sum_{d | gcd(q,n)} d * mu(q/d).  Integer."""
    g = gcd(q, n)
    return sum(d * mu[q // d] for d in divisors(g))


def ramanujan_trig(q, n):
    """Trigonometric definition (units summed over the multiplicative chart)."""
    return sum(np.cos(2 * np.pi * a * n / q) for a in range(1, q + 1) if gcd(a, q) == 1)


def cq_table(Qmax, Nmax, mu):
    """Matrix C[q, n] = c_q(n), 0..Qmax x 0..Nmax (Kluyver)."""
    C = np.zeros((Qmax + 1, Nmax + 1))
    for q in range(1, Qmax + 1):
        for n in range(1, Nmax + 1):
            C[q, n] = ramanujan(q, n, mu)
    return C


# ----------------------------------------------------------------------
# Chart latitude (discrete logarithm)
# ----------------------------------------------------------------------
def dlog_table(g, m):
    """lam[x] = discrete log base g of x in (Z/m)^x, for x = 1..m-1."""
    lam = [None] * m
    x = 1
    for k in range(m - 1):
        lam[x] = k
        x = (x * g) % m
    return lam


# ----------------------------------------------------------------------
# von Mangoldt, its convolution-inverse form, and the resonance
# ----------------------------------------------------------------------
def von_mangoldt(n):
    """True Lambda(n): log(prime) at prime powers, 0 otherwise."""
    if n < 2:
        return 0.0
    f = factorint(n)
    return float(log(min(f))) if len(f) == 1 else 0.0


def mobius_log(n, mu):
    """Convolution-inverse form (Thm 10.6b):  Lambda(n) = -sum_{d|n} mu(d) log d."""
    return -sum(mu[d] * log(d) for d in divisors(n) if d > 1)


def resonance_cumulative(Qmax, Nmax, mu, phi):
    """
    Return cum[L, n] = Lambda_L(n) = sum_{q<=L} mu(q)/phi(q) * c_q(n)
    for all L = 0..Qmax and n = 0..Nmax  (Mobius-Ramanujan partial sums, Def 10.5).
    """
    C = cq_table(Qmax, Nmax, mu)
    w = np.array([0.0] + [mu[q] / phi[q] if phi[q] else 0.0 for q in range(1, Qmax + 1)])
    return np.cumsum(w[:, None] * C, axis=0)


# ----------------------------------------------------------------------
# Full-orbit finite zeta   Z_P(k) = sum_{x in F_P^x} x^k   (in F_P)
# ----------------------------------------------------------------------
def finite_zeta(P, k):
    """Z_P(k) reduced into the representative range (-1 reported as P-1)."""
    return sum(pow(x, k, P) for x in range(1, P)) % P


# ----------------------------------------------------------------------
# F_{p^2} arithmetic:  element a + b*j  with  j^2 = d  (d a fixed nonresidue)
# ----------------------------------------------------------------------
def nonresidue(p):
    """Smallest quadratic nonresidue mod p."""
    for d in range(2, p):
        if pow(d, (p - 1) // 2, p) == p - 1:
            return d
    raise ValueError("no nonresidue found")


def fp2_mul(u, v, d, p):
    a, b = u; c, e = v
    return ((a * c + b * e * d) % p, (a * e + b * c) % p)


def fp2_pow(u, n, d, p):
    r = (1, 0)
    while n:
        if n & 1:
            r = fp2_mul(r, u, d, p)
        u = fp2_mul(u, u, d, p)
        n >>= 1
    return r


def fp2_conj(u, p):
    """Frobenius z -> z^p:  a + b j  ->  a - b j."""
    a, b = u
    return (a % p, (-b) % p)


def fp2_norm(u, d, p):
    a, b = u
    return (a * a - d * b * b) % p


def fp2_trace(u, p):
    a, b = u
    return (2 * a) % p


def fp2_order(u, d, p):
    """Multiplicative order of a unit u in F_{p^2}^x."""
    o = 1
    v = u
    while v != (1, 0):
        v = fp2_mul(v, u, d, p)
        o += 1
    return o


def sqrt_minus1(p, d):
    """Return a square root of -1 as an F_{p^2} element (a,b)."""
    # search the (small) field for (a,b)^2 = (-1, 0)
    for a in range(p):
        for b in range(p):
            if fp2_mul((a, b), (a, b), d, p) == ((p - 1) % p, 0) and (a, b) != (0, 0):
                return (a, b)
    raise ValueError("no sqrt(-1)")


def unit_circle(p, d):
    """The finite circle U_{p+1} = { z in F_{p^2}^x : N(z) = 1 }."""
    return [(a, b) for a in range(p) for b in range(p)
            if (a, b) != (0, 0) and fp2_norm((a, b), d, p) == 1]
