"""
=====================================================================
 CGPF - Constraint-Guided Probabilistic Forecaster   (was "BCP-PINN")
=====================================================================
"""
import os, numpy as np, pandas as pd, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------- SET THIS -------------------------
DATA_PATH = "/content/model_ready_v2.csv"
OUT_DIR   = "/content/"
MODE      = "validate"        # "validate" (tuning) or "test" (final, official weeks once)
USE_CONSTRAINTS = True        # ablation switch (constraints on/off)
N_SEEDS   = 5                 # ensemble size
PLOT_DMAS = ["DMA E","DMA C","DMA B"]; PLOT_WINDOW = 1
# --- validation weeks: earlier, seasonal spread, all BEFORE the first official week ---
VAL_STARTS = ["2021-08-02","2021-11-08","2022-01-24","2022-04-11"]
# ------------------------- config (defaults; FINALIZE ON VALIDATION) ---------------------------
LOOKBACK, HORIZON, STRIDE = 168, 168, 6
HIDDEN, EMB_DIM, LAYERS = 64, 8, 1
EPOCHS, PATIENCE, BATCH, LR, VAL_FRAC = 40, 6, 128, 1e-3, 0.15
CAL_SPAN_W = 8
LAM_Q, LAM_VOL, LAM_SMOOTH = 0.3, 0.02, 0.005
QUANTILES = [0.05,0.25,0.50,0.75,0.95]; Q_LO,Q_MED,Q_HI = 0,2,4
ALPHA = 0.10
CQR_COVER = 0.90              # nominal conformal level; TUNE on validation, not test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DMAS=[f"DMA {c}" for c in "ABCDEFGHIJ"]
WEATHER=["Rain","Temperature","Humidity","Windspeed"]
CTX=WEATHER+["hour_sin","hour_cos","wk_sin","wk_cos","holiday"]
LAGW=[168,336,504,672]

# ----------------------- data / evaluation -----------------------
def load_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Upload model_ready_v2.csv to Colab first.")
    m=pd.read_csv(DATA_PATH,index_col=0); m.index=pd.to_datetime(m.index,utc=True); return m

def test_windows(m):
    ev=m["eval_week"].values; b=[]; s=None
    for i,v in enumerate(ev):
        if v and s is None: s=i
        elif not v and s is not None: b.append((s,i)); s=None
    if s is not None: b.append((s,len(ev)))
    return b

def val_windows(m):
    out=[]
    for d in VAL_STARTS:
        ts=pd.Timestamp(d,tz="UTC"); idx=int(m.index.searchsorted(ts))
        out.append((idx, idx+HORIZON))
    return out

def get_windows(m): return val_windows(m) if MODE=="validate" else test_windows(m)

def pi_metrics(y,yhat):
    e=np.abs(np.asarray(y)-np.asarray(yhat)); return np.nanmean(e[:24]),np.nanmean(e[24:168]),np.nanmax(e[:24])
def prob_metrics(y,lo,hi,qall,taus):
    y=np.asarray(y,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float); ok=~np.isnan(y)
    cov=np.mean((y[ok]>=lo[ok])&(y[ok]<=hi[ok])); mpiw=np.mean(hi[ok]-lo[ok])
    pb=[np.mean(np.maximum(t*(y[ok]-qall[k][ok]),(t-1)*(y[ok]-qall[k][ok]))) for k,t in enumerate(taus)]
    w=(hi[ok]-lo[ok]).copy(); b=y[ok]<lo[ok]; a=y[ok]>hi[ok]
    w[b]+=2/ALPHA*(lo[ok][b]-y[ok][b]); w[a]+=2/ALPHA*(y[ok][a]-hi[ok][a])
    return cov,mpiw,float(np.mean(pb)),float(np.mean(w))

def seasonal_ref(x):
    r=np.full_like(x,np.nan)
    for t in range(len(x)):
        v=[x[t-L] for L in LAGW if t-L>=0]
        if v: r[t]=np.mean(v)
    r[np.isnan(r)]=np.nanmean(x); return r

# ---- per-window TRAINING-ONLY statistics (no leakage) ----
def fit_stats(m,origin):
    cal=origin-CAL_SPAN_W*168
    dstats={d:(float(np.nanmean(m[d].values[:cal])),float(np.nanstd(m[d].values[:cal])+1e-6)) for d in DMAS}
    wstats={w:(float(np.nanmean(m[w].values[:cal])),float(np.nanstd(m[w].values[:cal])+1e-6)) for w in WEATHER}
    return dstats,wstats

def scaled_ctx(m,wstats):
    base=m[CTX].values.astype(np.float32).copy()
    for j,w in enumerate(WEATHER):
        mu,sd=wstats[w]; base[:,j]=(base[:,j]-mu)/sd          # A3: scale weather, train stats
    return base

def build(m,lo,hi,dstats,wstats,refs):
    base=scaled_ctx(m,wstats); E,D,T,I,R=[],[],[],[],[]
    for di,d in enumerate(DMAS):
        mu,sd=dstats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd; ref=refs[d]
        first=max(0,lo-LOOKBACK); last=hi-HORIZON-LOOKBACK
        for s in range(first,last+1,STRIDE):
            ee=s+LOOKBACK; de=ee+HORIZON
            if ee<lo: continue
            E.append(np.column_stack([dem[s:ee],base[s:ee]])); D.append(base[ee:de])
            T.append(dem[ee:de]); I.append(di); R.append((np.mean(ref[ee:de])-mu)/sd)
    return (np.asarray(E,np.float32),np.asarray(D,np.float32),np.asarray(T,np.float32),
            np.asarray(I,np.int64),np.asarray(R,np.float32))

# ----------------------------- model -----------------------------
class CGPFNet(nn.Module):
    def __init__(s,ne,nd,nq):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.mu=nn.Linear(HIDDEN,1); s.dn=nn.Linear(HIDDEN,nq//2); s.up=nn.Linear(HIDDEN,nq//2)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        mu=s.mu(o); low=torch.flip(mu-torch.cumsum(F.softplus(s.dn(o)),-1),[-1]); high=mu+torch.cumsum(F.softplus(s.up(o)),-1)
        return torch.cat([low,mu,high],-1), mu

def point_loss(q,y):  return F.l1_loss(q[...,Q_MED],y)       # A3: MAE median (was MSE)
def band_loss(q,mu,y):
    qb=mu.detach()+(q-mu); L=0.0
    for k,t in enumerate(QUANTILES):
        if k==Q_MED: continue
        d=y-qb[...,k]; L=L+torch.mean(torch.maximum(t*d,(t-1)*d))
    return L
def constraint_loss(q,ref):
    md=q[...,Q_MED]; vol=torch.mean((md.mean(1)-ref)**2)
    d2=md[:,2:]-2*md[:,1:-1]+md[:,:-2]; return LAM_VOL*vol+LAM_SMOOTH*torch.mean(d2**2)

def within_dma_split(I):
    """A2: chronological split inside each DMA -> every DMA in train AND val."""
    tr=np.zeros(len(I),bool)
    for di in range(len(DMAS)):
        pos=np.where(I==di)[0]                      # already time-ordered
        cut=int(len(pos)*(1-VAL_FRAC)); tr[pos[:cut]]=True
    return tr

def train_window(m,origin,dstats,wstats,refs,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cal_lo=origin-CAL_SPAN_W*168
    E,D,T,I,R=build(m,0,cal_lo,dstats,wstats,refs)
    trm=within_dma_split(I)
    def dl(sel,sh): return DataLoader(TensorDataset(torch.tensor(E[sel]),torch.tensor(D[sel]),
        torch.tensor(T[sel]),torch.tensor(I[sel]),torch.tensor(R[sel])),batch_size=BATCH,shuffle=sh)
    tr,va=dl(trm,True),dl(~trm,False)
    net=CGPFNet(1+len(CTX),len(CTX),len(QUANTILES)-1).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for eb,db,tb,ib,rb in tr:
            eb,db,tb,ib,rb=[x.to(DEVICE) for x in (eb,db,tb,ib,rb)]; opt.zero_grad()
            q,mu=net(eb,db,ib); loss=point_loss(q,tb)+LAM_Q*band_loss(q,mu,tb)
            if USE_CONSTRAINTS: loss=loss+constraint_loss(q,rb)
            loss.backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for eb,db,tb,ib,rb in va:
                eb,db,tb,ib=[x.to(DEVICE) for x in (eb,db,tb,ib)]
                q,_=net(eb,db,ib); vs+=point_loss(q,tb.to(DEVICE)).item()*len(eb); vn+=len(eb)
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net

def cqr_corr(net,m,origin,dstats,wstats,refs):
    cal_lo=origin-CAL_SPAN_W*168; E,D,T,I,R=build(m,cal_lo,origin,dstats,wstats,refs); corr={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            sel=I==di
            if sel.sum()==0: corr[d]=0.0; continue
            mu,sd=dstats[d]; q,_=net(torch.tensor(E[sel]).to(DEVICE),torch.tensor(D[sel]).to(DEVICE),torch.tensor(I[sel]).to(DEVICE))
            q=q.cpu().numpy(); lo=q[...,Q_LO]*sd+mu; hi=q[...,Q_HI]*sd+mu; y=T[sel]*sd+mu
            Esc=np.maximum(lo-y,y-hi).ravel(); nE=len(Esc); lv=min(1.0,np.ceil((nE+1)*CQR_COVER)/nE)
            corr[d]=float(np.quantile(Esc,lv))
    return corr

def predict(net,m,origin,dstats,wstats,corr):
    base=scaled_ctx(m,wstats); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=dstats[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[origin-LOOKBACK:origin],base[origin-LOOKBACK:origin]])[None]
            dec=base[origin:origin+HORIZON][None]
            q,_=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]).to(DEVICE))
            q=q.cpu().numpy()[0]*sd+mu; qa=[np.clip(q[:,k],0,None) for k in range(len(QUANTILES))]
            lo=np.clip(q[:,Q_LO]-corr[d],0,None); hi=np.clip(q[:,Q_HI]+corr[d],0,None)
            out[d]=dict(median=qa[Q_MED],lo=lo,hi=hi,qall=qa)
    return out

def forecast_all(m,wins,refs,seed):
    out={}
    for wi,(s,e) in enumerate(wins,1):
        ds,ws=fit_stats(m,s); net=train_window(m,s,ds,ws,refs,seed)
        corr=cqr_corr(net,m,s,ds,ws,refs); out[wi]=(s,e,predict(net,m,s,ds,ws,corr))
    return out

def ensemble(runs):
    ens={}
    for wi in runs[0]:
        s,e,_=runs[0][wi]; pe={}
        for d in DMAS:
            pe[d]=dict(median=np.mean([r[wi][2][d]["median"] for r in runs],0),
                       lo=np.mean([r[wi][2][d]["lo"] for r in runs],0),
                       hi=np.mean([r[wi][2][d]["hi"] for r in runs],0),
                       qall=[np.mean([r[wi][2][d]["qall"][k] for r in runs],0) for k in range(len(QUANTILES))])
        ens[wi]=(s,e,pe)
    return ens

def score(m,pred):
    rows=[]
    for wi,(s,e,p) in pred.items():
        for d in DMAS:
            y=m[d].iloc[s:e].values.astype(float).copy()
            imp=m[d+"_imp"].iloc[s:e].values.astype(bool); y[imp]=np.nan; P=p[d]
            pi1,pi2,pi3=pi_metrics(y,P["median"]); cov,mpiw,pb,wk=prob_metrics(y,P["lo"],P["hi"],P["qall"],QUANTILES)
            rows.append(dict(window=str(m.index[s].date()),dma=d,PI1=pi1,PI2=pi2,PI3=pi3,
                             COV90=cov,MPIW=mpiw,Pinball=pb,Winkler=wk))
    return pd.DataFrame(rows)

if __name__=="__main__":
    t0=time.time(); m=load_model(); wins=get_windows(m)
    refs={d:seasonal_ref(m[d].values.astype(np.float32)) for d in DMAS}
    print(f"Device:{DEVICE} | MODE={MODE} | constraints={USE_CONSTRAINTS} | seeds={N_SEEDS} | {len(wins)} weeks")
    for wi,(s,e) in enumerate(wins,1): print(f"  window {wi}: {m.index[s]} -> {m.index[e-1]}")
    runs=[]
    for si in range(N_SEEDS):
        seed=42+si; r=forecast_all(m,wins,refs,seed); runs.append(r)
        ss=score(m,r)[["PI1","COV90","Winkler"]].mean()
        print(f"  [seed {seed}] PI1={ss.PI1:.3f} COV90={ss.COV90:.3f} Winkler={ss.Winkler:.3f}")
    detail=score(m,ensemble(runs)); summ=detail[["PI1","PI2","PI3","COV90","MPIW","Pinball","Winkler"]].mean()
    print(f"\n=== CGPF ENSEMBLE ({MODE}, {N_SEEDS} seeds, mean over {len(wins)} weeks x 10 DMAs) ===")
    print(summ.round(4).to_string())
    tag=MODE+("_cons" if USE_CONSTRAINTS else "_nocons")
    detail.to_csv(OUT_DIR+f"cgpf_{tag}_detail.csv",index=False)
    summ.to_frame().T.to_csv(OUT_DIR+f"cgpf_{tag}_summary.csv",index=False)
    detail.groupby("dma")[["PI1","COV90","Winkler"]].mean().round(3).to_csv(OUT_DIR+f"cgpf_{tag}_perzone.csv")
    print(f"\nSaved -> {OUT_DIR}cgpf_{tag}_*.csv   ({time.time()-t0:.0f}s)")

    # ---- plot (inline, not saved) ----
    ens=ensemble(runs); s,e,pred=ens[PLOT_WINDOW]; hours=np.arange(HORIZON)
    fig,axes=plt.subplots(len(PLOT_DMAS),1,figsize=(11,2.7*len(PLOT_DMAS)),sharex=True)
    if len(PLOT_DMAS)==1: axes=[axes]
    for ax,d in zip(axes,PLOT_DMAS):
        y=m[d].iloc[s:e].values.astype(float); P=pred[d]
        ax.fill_between(hours,P["lo"],P["hi"],color="#fdae61",alpha=0.45,label="90% calibrated band")
        ax.plot(hours,y,color="#1b9e77",lw=1.7,label="Actual")
        ax.plot(hours,P["median"],color="#d7191c",lw=1.5,ls="--",label="Median forecast")
        ax.axvspan(0,24,color="#2c7bb6",alpha=0.07); ax.set_ylabel(f"{d}\n(L/s)"); ax.grid(alpha=0.25); ax.legend(loc="upper right",fontsize=8)
    axes[-1].set_xlabel("Forecast hour (0-168)")
    fig.suptitle(f"CGPF ({MODE}) - calibrated probabilistic forecast (window {PLOT_WINDOW})",y=1.0,fontsize=11)
    plt.tight_layout(); plt.show()
