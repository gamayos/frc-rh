import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import frc_core as fc
mp.mp.dps = 18

INK="#1a1a1a"; STAIR="#111111"; GRIDC="#d9d9d9"
# light->dark gradient as more modes are added
CL = ["#f0a8a8","#df6b6b","#c0392b","#7d1f15"]    # classical (reds)
FR = ["#a8c4ec","#5f8fd6","#1f5fbf","#143f86"]    # FRC (blues)
plt.rcParams.update({"font.size":10,"axes.edgecolor":INK,"axes.linewidth":0.8})

X0, X1 = 2, 30
xs = np.linspace(X0+0.02, X1, 1400)
ns = np.arange(X0, X1+1)
ispp = np.array([fc.von_mangoldt(n)>0 for n in ns])
psi_true = np.cumsum([fc.von_mangoldt(n) for n in ns])      # Chebyshev staircase

# ---------- classical side ----------
M_list=[5,20,80]
gam=[mp.zetazero(m) for m in range(1,max(M_list)+1)]        # rho_m = 1/2 + i gamma_m
def psi_M(x,M):
    s=mp.mpf(0)
    for r in gam[:M]:
        s += 2*mp.re(mp.power(x, r)/r)
    return float(x - mp.log(2*mp.pi) - 0.5*mp.log(1-x**-2) - s)

# ---------- FRC side ----------
Qmax=300; mu,phi=fc.sieve_mu_phi(Qmax)
cum=fc.resonance_cumulative(Qmax,X1,mu,phi)                 # cum[L,n]=R_L(n)
L_list=[15,60,240]
def Psi_L(L):                                               # (n/phi) R_L cumulated -> psi
    w=np.array([(n/phi[n])*cum[L,n] for n in ns])
    return np.cumsum(w)

fig,ax=plt.subplots(2,2,figsize=(13.5,8.2))

# ===== TOP-LEFT : classical spectral modes (zeros) =====
A=ax[0,0]
for i,m in enumerate([1,2,3]):
    r=gam[m-1]
    g=np.array([float(-2*mp.re(mp.power(x,r)/r)) for x in xs])
    A.plot(xs,g,color=CL[i],lw=1.4,label=r"$\rho_{%d}=0.5+i\,%.2f$" % (m, float(mp.im(r))))
A.axhline(0,color=GRIDC,lw=0.8)
A.set_title("Classical spectral modes: the zeros"); A.set_xlabel("$x$")
A.set_ylabel(r"$-2\,\mathrm{Re}\,x^{\rho_m}/\rho_m$"); A.legend(fontsize=8,loc="upper left"); A.set_xlim(X0,X1)

# ===== TOP-RIGHT : FRC spectral modes (units-chart, energy order) =====
B=ax[0,1]
for i,q in enumerate([2,3,5]):
    cq=np.array([fc.ramanujan(q,int(n),mu) for n in ns])
    w=mu[q]/phi[q]
    B.step(ns, w*cq, where="mid", color=FR[i], lw=1.5,
           label=fr"$q={q}$  (energy ${mu[q]**2/phi[q]:.2f}$)")
B.axhline(0,color=GRIDC,lw=0.8)
B.set_title("FRC spectral modes: units-chart $\\frac{\\mu(q)}{\\varphi(q)}c_q$  (antipode $q{=}2$ first)")
B.set_xlabel("$n$"); B.set_ylabel(r"$\frac{\mu(q)}{\varphi(q)}\,c_q(n)$")
B.legend(fontsize=8,loc="upper right"); B.set_xlim(X0,X1)

# ===== BOTTOM-LEFT : classical emergence of psi(x) =====
C=ax[1,0]
C.step(np.concatenate([[X0],ns]), np.concatenate([[0],psi_true]), where="post",
       color=STAIR, lw=2.0, label=r"$\psi(x)$ (target)")
for i,M in enumerate(M_list):
    y=np.array([psi_M(x,M) for x in xs])
    C.plot(xs,y,color=CL[i],lw=1.3,label=f"{M} zeros")
for n in ns[ispp]: C.axvline(n,color=GRIDC,lw=0.7,ls=":")
C.set_title("Classical: $\\psi(x)$ from $M$ zeros"); C.set_xlabel("$x$")
C.set_ylabel(r"$\psi_M(x)$"); C.legend(fontsize=8,loc="upper left"); C.set_xlim(X0,X1); C.set_ylim(0,X1+4)

# ===== BOTTOM-RIGHT : FRC emergence of the staircase =====
D=ax[1,1]
D.step(np.concatenate([[X0],ns]), np.concatenate([[0],psi_true]), where="post",
       color=STAIR, lw=2.0, label=r"$\psi(N)$ (target)")
for i,L in enumerate(L_list):
    D.plot(ns, Psi_L(L), color=FR[i], lw=1.4, marker="", label=fr"bandwidth $L={L}$")
for n in ns[ispp]: D.axvline(n,color=GRIDC,lw=0.7,ls=":")
D.set_title(r"FRC: staircase from $L$ units-chart modes, $\Psi_L(N)=\sum_{n\leq N}\frac{n}{\varphi(n)}R_L(n)$")
D.set_xlabel(r"$N$  (prime meridian, read through horizon $\sqrt{p}$)")
D.set_ylabel(r"$\Psi_L(N)$"); D.legend(fontsize=8,loc="upper left"); D.set_xlim(X0,X1); D.set_ylim(0,X1+4)

fig.suptitle("Prime-counting function from spectral modes:  classical zeros  (left)   $\\;\\longleftrightarrow\\;$   FRC units-chart modes  (right)",
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig("fig_emergence_frc.pdf",bbox_inches="tight"); print("saved fig_emergence_frc.pdf")
