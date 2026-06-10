import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
mp.mp.dps = 18

INK="#1a1a1a"; SPA="#1f5fbf"; FIN="#c0392b"; GRID="#cccccc"
plt.rcParams.update({"font.size":10,"axes.edgecolor":INK,"axes.linewidth":0.8})

def Z_true(t): return float(mp.siegelz(t))
def Z_horizon(t):
    N=int(mp.floor(mp.sqrt(t/(2*mp.pi))))            # = FRC horizon sqrt(p), p=t/2pi
    th=mp.siegeltheta(t)
    return float(2*mp.fsum(mp.cos(th - t*mp.log(n))/mp.sqrt(n) for n in range(1,N+1)))

fig,ax=plt.subplots(1,3,figsize=(15,4.3))

# -- Panel A: finite-horizon reconstruction reproduces the zeros --
ts=np.linspace(10,55,700)
zt=[Z_true(t) for t in ts]; zh=[Z_horizon(t) for t in ts]
gam=[float(mp.zetazero(k).imag) for k in range(1,11)]
ax[0].axhline(0,color=GRID,lw=0.8)
for g in gam: ax[0].axvline(g,color=GRID,lw=0.8,ls=":")
ax[0].plot(ts,zt,color=INK,lw=1.6,label=r"true $Z(t)$")
ax[0].plot(ts,zh,color=FIN,lw=1.4,ls="--",label=r"horizon sum, $N=\lfloor\sqrt{t/2\pi}\rfloor$")
ax[0].plot(gam,[0]*len(gam),"o",color=SPA,ms=5,zorder=5,label=r"zeros $\gamma_n$")
ax[0].set_xlabel(r"height $t$  ($=2\pi p$ at shell scale $p$)"); ax[0].set_ylabel(r"$Z(t)$")
ax[0].set_title("De-framing = finite Riemann--Siegel:\nhorizon data reconstructs the zeros")
ax[0].legend(fontsize=8,loc="upper right"); ax[0].set_xlim(10,55); ax[0].set_ylim(-6,6)

# -- Panel B: zero density = shell scale-depth (1/2pi) log p --
ps=np.logspace(1.3,5,30)
dens=[float((1/(2*mp.pi))*mp.log(p)) for p in ps]          # (1/2pi) log p
densT=[float((1/(2*mp.pi))*mp.log((2*mp.pi*p)/(2*mp.pi))) for p in ps]  # (1/2pi) log(T/2pi), T=2pi p
ax[1].plot(ps,densT,color=INK,lw=1.8,label=r"R--vM density $\frac{1}{2\pi}\log\frac{T}{2\pi}$, $T=2\pi p$")
ax[1].plot(ps,dens,color=FIN,lw=0,marker="o",ms=4,label=r"shell scale-depth $\frac{1}{2\pi}\log p$")
ax[1].set_xscale("log"); ax[1].set_xlabel(r"shell cardinality $p$")
ax[1].set_ylabel(r"zeros per unit height"); ax[1].legend(fontsize=8.5,loc="upper left")
ax[1].set_title("Zero density $=$ shell scale-depth\n(exact: archimedean $\\log$ is the shell octave-count)")

# -- Panel C: reconstruction error = horizon scale p^{-1/4} (unconditional) --
T0=np.array([40,80,160,320,640,1280,2560,5120]); err=[]
for t0 in T0:
    e=[abs(Z_horizon(float(t0)+0.123*i)-Z_true(float(t0)+0.123*i)) for i in range(1,16)]
    err.append(sorted(e)[len(e)//2])
ax[2].loglog(T0,err,"o-",color=FIN,lw=1.4,ms=5,label=r"median $|Z_{\rm horizon}-Z|$")
ax[2].loglog(T0,[float(t**-0.25) for t in T0],color=INK,lw=1.6,ls="--",label=r"$t^{-1/4}=p^{-1/4}$")
ax[2].set_xlabel(r"height $t$  ($=2\pi p$)"); ax[2].set_ylabel("reconstruction error")
ax[2].legend(fontsize=8.5,loc="lower left")
ax[2].set_title("Residue $=$ horizon-wrapping scale $p^{-1/4}$\n(unconditional: no GRH used)")

plt.tight_layout(); plt.savefig("fig_deframing.pdf",bbox_inches="tight"); print("saved fig_deframing.pdf")
