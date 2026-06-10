import numpy as np, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
mp.mp.dps=20
INK="#1a1a1a"; ZER="#1f5fbf"; RMT="#16a085"; SEMI="#c0392b"; GRIDC="#d9d9d9"
plt.rcParams.update({"font.size":10,"axes.edgecolor":INK,"axes.linewidth":0.8})

g=np.load('/tmp/zeros.npy'); u=np.load('/tmp/unfolded.npy')
s=np.diff(u); s/=s.mean()

fig,ax=plt.subplots(1,2,figsize=(13,4.6))

# --- Panel A: GUE level repulsion (the RMT side) ---
A=ax[0]
A.hist(s,bins=np.linspace(0,3,16),density=True,color=ZER,alpha=0.55,edgecolor="white",label=f"zeta zeros (first {len(g)})")
xx=np.linspace(0,3,400)
A.plot(xx,(32/math.pi**2)*xx**2*np.exp(-4*xx**2/math.pi),color=RMT,lw=2.2,label="GUE (random matrix)")
A.plot(xx,np.exp(-xx),color=INK,lw=1.4,ls="--",label="Poisson (uncorrelated)")
A.set_xlabel("normalized spacing  $s$"); A.set_ylabel("$P(s)$")
A.set_title(r"$\sigma(\hat H)$ shows level repulsion: GUE, not Poisson"+"\n"+r"(eigenvalues of a self-adjoint operator)")
A.legend(fontsize=8.5); A.set_xlim(0,3)

# --- Panel B: spectral counting = Berry-Keating semiclassical = scale-depth ---
B=ax[1]
Ts=np.linspace(10,450,60)
Nact=np.array([int(mp.nzeros(float(T))) for T in Ts])
Nsemi=(Ts/(2*math.pi))*np.log(Ts/(2*math.pi)) - Ts/(2*math.pi) + 7/8
B.step(np.concatenate([[10],g]),np.arange(len(g)+1),where="post",color=ZER,lw=1.4,label=r"eigenvalue count $N(T)=\#\{\gamma_n\leq T\}$")
B.plot(Ts,Nsemi,color=SEMI,lw=2.0,ls="--",label=r"Berry--Keating $\frac{T}{2\pi}\!\left(\log\frac{T}{2\pi}-1\right)$")
B.set_xlabel(r"eigenvalue $T=\gamma$  $(=2\pi p$ at shell $p)$"); B.set_ylabel(r"$N(T)$")
B.set_title(r"density $\frac{dN}{dT}=\frac{1}{2\pi}\log\frac{T}{2\pi}=\frac{1}{2\pi}\log p$"+"\n"+r"(shell scale-depth $=$ Planck cutoff $\hbar$)")
B.legend(fontsize=8.5,loc="upper left"); B.set_xlim(10,450)
# annotate scale-depth at one point
B.axvline(2*math.pi*50,color=GRIDC,lw=0.8,ls=":")
B.annotate(r"$p\!=\!50:\ T\!=\!2\pi p,\ \rho_{\rm density}=\frac{\log 50}{2\pi}$",
           xy=(2*math.pi*50,55),xytext=(2*math.pi*50+20,30),fontsize=8,color=INK,
           arrowprops=dict(arrowstyle="->",color=GRIDC))

fig.suptitle(r"The zeros as eigenvalues of $\hat H$ (scale-time evolution generator):  $\sigma(\hat H)=\{-i(\rho-\frac{1}{2})\}$",
             fontsize=12,y=1.0)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("fig_hilbert_polya.pdf",bbox_inches="tight"); print("saved fig_hilbert_polya.pdf")
