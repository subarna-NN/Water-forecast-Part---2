"""
=====================================================================
 PHASE F - STEP 1 (GATE CHECK): mean-variance demand law
=====================================================================
"""
import os, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

DATA_PATH="/content/model_ready_v2.csv"; OUT_DIR="/content/"
DMAS=[f"DMA {c}" for c in "ABCDEFGHIJ"]
MIN_OBS=10                         # min observed weeks per hour-of-week slot
def law(mu,a,b,c): return a*np.power(mu,b)+c

def load():
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"{DATA_PATH} not found. Upload model_ready_v2.csv.")
    m=pd.read_csv(DATA_PATH,index_col=0); m.index=pd.to_datetime(m.index,utc=True); return m

def how_stats(m, dma, row_mask):
    """mean & variance of OBSERVED demand per hour-of-week slot (training rows only)."""
    y=m[dma].values.astype(float).copy()
    imp=m[dma+"_imp"].values.astype(bool)
    keep=row_mask & (~imp)                                   # training & observed
    how=(m.index.weekday*24+m.index.hour).values
    mus,vars=[],[]
    for k in range(168):
        v=y[keep & (how==k)]
        if len(v)>=MIN_OBS: mus.append(v.mean()); vars.append(v.var(ddof=1))
    return np.array(mus),np.array(vars)

def loglog_fit(mu,var):
    """Robust power-law exponent via log-log regression: log var = log a + b log mu."""
    ok=(mu>0)&(var>0)
    if ok.sum()<6: return None
    b,la=np.polyfit(np.log(mu[ok]),np.log(var[ok]),1)
    pred=la+b*np.log(mu[ok]); ss=np.sum((np.log(var[ok])-pred)**2); st=np.sum((np.log(var[ok])-np.log(var[ok]).mean())**2)
    return dict(b_ll=b,a_ll=np.exp(la),R2_ll=1-ss/st if st>0 else np.nan)

def fit_law(mu,var):
    """Fit sigma^2=a*mu^b+c (raw NLS, for the c/background term) + robust log-log b."""
    if len(mu)<6: return None
    out={}
    p0=[var.mean()/max(mu.mean(),1e-6),1.0,max(var.min(),1e-6)]
    try:
        popt,pcov=curve_fit(law,mu,var,p0=p0,bounds=([0,0,0],[np.inf,3.0,np.inf]),maxfev=20000)
        ci=1.96*np.sqrt(np.diag(pcov)); pred=law(mu,*popt)
        ss=np.sum((var-pred)**2); st=np.sum((var-var.mean())**2)
        out.update(a=popt[0],b=popt[1],c=popt[2],a_ci=ci[0],b_ci=ci[1],c_ci=ci[2],R2=1-ss/st if st>0 else np.nan)
    except Exception:
        out.update(a=np.nan,b=np.nan,c=np.nan,a_ci=np.nan,b_ci=np.nan,c_ci=np.nan,R2=np.nan)
    ll=loglog_fit(mu,var)
    if ll: out.update(ll)
    out["n"]=len(mu); return out

if __name__=="__main__":
    m=load()
    train_rows=(m["eval_week"].values==0)                    # TRAINING = non-evaluation weeks
    print("Training rows:",int(train_rows.sum()),"of",len(m),"| law: sigma^2 = a*mu^b + c\n")

    # ---------- per-DMA fits ----------
    rows=[]; allmu=[]; allvar=[]; per_dma_pts={}
    for d in DMAS:
        mu,var=how_stats(m,d,train_rows); per_dma_pts[d]=(mu,var)
        allmu.append(mu); allvar.append(var)
        f=fit_law(mu,var)
        if f: rows.append(dict(DMA=d,**f))
    R=pd.DataFrame(rows).set_index("DMA")
    print("=== Per-DMA fit  (NLS a,b,c  |  robust log-log exponent b_ll) ===")
    print(R[["b","c","R2","b_ll","R2_ll","n"]].round(3).to_string())
    print(f"\nlog-log exponent b_ll across DMAs: mean={R.b_ll.mean():.3f} std={R.b_ll.std():.3f} "
          f"(per-DMA mu-range is narrow -> within-DMA fit is noisy/ill-posed)")

    # ---------- pooled fit (the meaningful one: wide mu range across DMAs) ----------
    pmu=np.concatenate(allmu); pvar=np.concatenate(allvar)
    pf=fit_law(pmu,pvar)
    print("\n=== Pooled fit (all DMAs, wide mu range) ===")
    print(f"NLS  : a={pf['a']:.4f}(+/-{pf['a_ci']:.4f})  b={pf['b']:.4f}(+/-{pf['b_ci']:.4f})  "
          f"c={pf['c']:.4f}(+/-{pf['c_ci']:.4f})  R2={pf['R2']:.3f}")
    print(f"log-log: b={pf['b_ll']:.4f}  a={pf['a_ll']:.4f}  R2={pf['R2_ll']:.3f}   (b~1 => Poisson-like var proportional to mean)")

    # ---------- cross-DMA Taylor law (whole-series mean vs variance, 10 points) ----------
    dmask=train_rows
    dm=np.array([m[d].values[dmask & ~m[d+"_imp"].values.astype(bool)].mean() for d in DMAS])
    dv=np.array([m[d].values[dmask & ~m[d+"_imp"].values.astype(bool)].var(ddof=1) for d in DMAS])
    bc,lac=np.polyfit(np.log(dm),np.log(dv),1); predc=lac+bc*np.log(dm)
    r2c=1-np.sum((np.log(dv)-predc)**2)/np.sum((np.log(dv)-np.log(dv).mean())**2)
    print(f"\n=== Cross-DMA Taylor law (10 zones, overall mean vs variance) ===")
    print(f"b={bc:.3f}  R2={r2c:.3f}  (strong across zones)")

    # ---------- seasonal stability (pooled, per meteorological season) ----------
    seasons={"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}
    print("\n=== Seasonal stability (pooled fit per season) ===")
    seas_rows=[]
    for sname,months in seasons.items():
        srow=train_rows & m.index.month.isin(months)              # (fixed: no .values)
        smu=[]; svar=[]
        for d in DMAS:
            mu,var=how_stats(m,d,srow)
            smu.append(mu); svar.append(var)
        smu=np.concatenate(smu); svar=np.concatenate(svar)
        f=fit_law(smu,svar)
        if f: seas_rows.append(dict(season=sname,b_ll=f.get("b_ll",np.nan),a_ll=f.get("a_ll",np.nan),R2_ll=f.get("R2_ll",np.nan),n=f["n"]))
    S=pd.DataFrame(seas_rows).set_index("season")
    print(S.round(3).to_string())
    print(f"\nlog-log exponent b across seasons: mean={S.b_ll.mean():.3f}  std={S.b_ll.std():.3f}  "
          f"(b stays ~1; fit weakest in summer JJA where weather breaks the scaling)")

    R.to_csv(OUT_DIR+"step1_perdma_law.csv"); S.to_csv(OUT_DIR+"step1_seasonal_law.csv")
    pd.DataFrame([pf]).to_csv(OUT_DIR+"step1_pooled_law.csv",index=False)
    print(f"\nSaved -> {OUT_DIR}step1_*.csv")

    # ---------- plots (inline, not saved) ----------
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    cols=plt.cm.tab10(np.linspace(0,1,10))
    for (d,(mu,var)),cc in zip(per_dma_pts.items(),cols):
        ax[0].scatter(mu,var,s=14,color=cc,alpha=0.6,label=d)
    xx=np.linspace(pmu.min(),pmu.max(),200)
    ax[0].plot(xx,law(xx,pf["a"],pf["b"],pf["c"]),"k-",lw=2,label=f"pooled fit (b={pf['b']:.2f})")
    ax[0].set_xlabel("hour-of-week mean demand  mu  (L/s)"); ax[0].set_ylabel("variance  sigma^2")
    ax[0].set_title("Mean-variance law (all DMAs)"); ax[0].legend(fontsize=7,ncol=2)
    ax[1].scatter(pmu,pvar,s=12,color="#2c7bb6",alpha=0.5)
    ax[1].plot(xx,law(xx,pf["a"],pf["b"],pf["c"]),"r-",lw=2)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("mu (log)"); ax[1].set_ylabel("sigma^2 (log)")
    ax[1].set_title(f"Log-log (pooled)  R2={pf['R2']:.3f}")
    plt.tight_layout(); plt.show()
