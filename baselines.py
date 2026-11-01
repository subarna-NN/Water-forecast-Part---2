"""
=====================================================================
 PHASE B - Final official-week comparison (TEST, reported ONCE)
=====================================================================
"""
import os, warnings, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")                       # silence all-imputed-slice notices (cosmetic)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import wilcoxon
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_PATH="/content/model_ready_v2.csv"; OUT_DIR="/content/"
N_SEEDS=5
# ---- frozen config (chosen on validation) ----
LOOKBACK,HORIZON,STRIDE=168,168,6
HIDDEN,EMB_DIM,LAYERS=64,8,1
EPOCHS,PATIENCE,BATCH,LR,VAL_FRAC=40,6,128,1e-3,0.15
CAL_SPAN_W=8
LAM_Q,LAM_VOL,LAM_SMOOTH=0.3,0.02,0.005
QUANTILES=[0.05,0.25,0.50,0.75,0.95]; Q_LO,Q_MED,Q_HI=0,2,4
ALPHA=0.10; CQR_COVER=0.91
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DMAS=[f"DMA {c}" for c in "ABCDEFGHIJ"]
WEATHER=["Rain","Temperature","Humidity","Windspeed"]
CTX=WEATHER+["hour_sin","hour_cos","wk_sin","wk_cos","holiday"]
LAGW=[168,336,504,672]
XGB_FEATS=WEATHER+["hour_sin","hour_cos","wk_sin","wk_cos","month","holiday"]

def load_model():
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"{DATA_PATH} not found. Upload model_ready_v2.csv.")
    m=pd.read_csv(DATA_PATH,index_col=0); m.index=pd.to_datetime(m.index,utc=True); return m
def test_windows(m):
    ev=m["eval_week"].values; b=[]; s=None
    for i,v in enumerate(ev):
        if v and s is None: s=i
        elif not v and s is not None: b.append((s,i)); s=None
    if s is not None: b.append((s,len(ev)))
    return b
def pi_metrics(y,yh):
    e=np.abs(np.asarray(y)-np.asarray(yh)); ok=~np.isnan(e)
    d1=e[:24][~np.isnan(e[:24])]; d2=e[24:168][~np.isnan(e[24:168])]
    pi1=np.mean(d1) if len(d1) else np.nan; pi2=np.mean(d2) if len(d2) else np.nan
    pi3=np.max(d1) if len(d1) else np.nan; return pi1,pi2,pi3
def prob_metrics(y,lo,hi,qall,taus):
    y=np.asarray(y,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float); ok=~np.isnan(y)
    if ok.sum()==0: return (np.nan,)*4
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
def fit_stats(m,origin):
    cal=origin-CAL_SPAN_W*168
    d={x:(float(np.nanmean(m[x].values[:cal])),float(np.nanstd(m[x].values[:cal])+1e-6)) for x in DMAS}
    w={x:(float(np.nanmean(m[x].values[:cal])),float(np.nanstd(m[x].values[:cal])+1e-6)) for x in WEATHER}
    return d,w
def scaled_ctx(m,wstats):
    base=m[CTX].values.astype(np.float32).copy()
    for j,w in enumerate(WEATHER): mu,sd=wstats[w]; base[:,j]=(base[:,j]-mu)/sd
    return base
def build(m,lo,hi,ds,ws,refs):
    base=scaled_ctx(m,ws); E,D,T,I,R=[],[],[],[],[]
    for di,d in enumerate(DMAS):
        mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd; ref=refs[d]
        first=max(0,lo-LOOKBACK); last=hi-HORIZON-LOOKBACK
        for s in range(first,last+1,STRIDE):
            ee=s+LOOKBACK; de=ee+HORIZON
            if ee<lo: continue
            E.append(np.column_stack([dem[s:ee],base[s:ee]])); D.append(base[ee:de])
            T.append(dem[ee:de]); I.append(di); R.append((np.mean(ref[ee:de])-mu)/sd)
    return (np.asarray(E,np.float32),np.asarray(D,np.float32),np.asarray(T,np.float32),
            np.asarray(I,np.int64),np.asarray(R,np.float32))

# ---------------- neural model (shared backbone) ----------------
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
def point_loss(q,y): return F.l1_loss(q[...,Q_MED],y)
def band_loss(q,mu,y):
    qb=mu.detach()+(q-mu); L=0.0
    for k,t in enumerate(QUANTILES):
        if k==Q_MED: continue
        d=y-qb[...,k]; L=L+torch.mean(torch.maximum(t*d,(t-1)*d))
    return L
def constraint_loss(q,r):
    md=q[...,Q_MED]; v=torch.mean((md.mean(1)-r)**2); d2=md[:,2:]-2*md[:,1:-1]+md[:,:-2]
    return LAM_VOL*v+LAM_SMOOTH*torch.mean(d2**2)
def within_dma_split(I):
    tr=np.zeros(len(I),bool)
    for di in range(len(DMAS)):
        pos=np.where(I==di)[0]; cut=int(len(pos)*(1-VAL_FRAC)); tr[pos[:cut]]=True
    return tr
def train_nn(m,origin,ds,ws,refs,seed,use_band,use_cons):
    torch.manual_seed(seed); np.random.seed(seed)
    E,D,T,I,R=build(m,0,origin-CAL_SPAN_W*168,ds,ws,refs); trm=within_dma_split(I)
    def dl(sel,sh): return DataLoader(TensorDataset(torch.tensor(E[sel]),torch.tensor(D[sel]),
        torch.tensor(T[sel]),torch.tensor(I[sel]),torch.tensor(R[sel])),batch_size=BATCH,shuffle=sh)
    tr,va=dl(trm,True),dl(~trm,False)
    net=CGPFNet(1+len(CTX),len(CTX),len(QUANTILES)-1).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR)
    best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for eb,db,tb,ib,rb in tr:
            eb,db,tb,ib,rb=[x.to(DEVICE) for x in (eb,db,tb,ib,rb)]; opt.zero_grad()
            q,mu=net(eb,db,ib); loss=point_loss(q,tb)
            if use_band: loss=loss+LAM_Q*band_loss(q,mu,tb)
            if use_cons: loss=loss+constraint_loss(q,rb)
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
def cqr_corr(net,m,origin,ds,ws,refs):
    E,D,T,I,R=build(m,origin-CAL_SPAN_W*168,origin,ds,ws,refs); corr={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            sel=I==di
            if sel.sum()==0: corr[d]=0.0; continue
            mu,sd=ds[d]; q,_=net(torch.tensor(E[sel]).to(DEVICE),torch.tensor(D[sel]).to(DEVICE),torch.tensor(I[sel]).to(DEVICE))
            q=q.cpu().numpy(); lo=q[...,Q_LO]*sd+mu; hi=q[...,Q_HI]*sd+mu; y=T[sel]*sd+mu
            Es=np.maximum(lo-y,y-hi).ravel(); nE=len(Es); lv=min(1.0,np.ceil((nE+1)*CQR_COVER)/nE)
            corr[d]=float(np.quantile(Es,lv))
    return corr
def predict_nn(net,m,origin,ds,ws,corr,use_band):
    base=scaled_ctx(m,ws); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[origin-LOOKBACK:origin],base[origin-LOOKBACK:origin]])[None]
            dec=base[origin:origin+HORIZON][None]
            q,_=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]).to(DEVICE))
            q=q.cpu().numpy()[0]*sd+mu; med=np.clip(q[:,Q_MED],0,None)
            if use_band:
                qa=[np.clip(q[:,k],0,None) for k in range(len(QUANTILES))]
                lo=np.clip(q[:,Q_LO]-corr[d],0,None); hi=np.clip(q[:,Q_HI]+corr[d],0,None)
                out[d]=dict(median=med,lo=lo,hi=hi,qall=qa)
            else:
                out[d]=dict(median=med,lo=None,hi=None,qall=None)
    return out

# ---------------- point baselines ----------------
def naive_pred(m,origin):
    out={}
    for d in DMAS:
        s=m[d].values.astype(float); p=np.empty(HORIZON)
        for h in range(HORIZON):
            v=[s[origin+h-168*k] for k in range(1,5) if origin+h-168*k>=0]
            p[h]=np.nanmean(v) if v else np.nan
        out[d]=dict(median=np.clip(p,0,None),lo=None,hi=None,qall=None)
    return out
def xgb_pred(m,origin):
    out={}
    for d in DMAS:
        s=pd.Series(m[d].values.astype(float),index=m.index); X=m[XGB_FEATS].copy()
        for L in LAGW: X[f"l{L}"]=s.shift(L)
        X["lm"]=pd.concat([s.shift(L) for L in LAGW],axis=1).mean(axis=1)
        tr=np.zeros(len(m),bool); tr[:origin]=True; val=X.notna().all(axis=1).values&tr
        mdl=xgb.XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
                             colsample_bytree=0.8,random_state=42,n_jobs=4,objective="reg:squarederror")
        mdl.fit(X[val],s[val]); p=mdl.predict(X.iloc[origin:origin+HORIZON])
        out[d]=dict(median=np.clip(p,0,None),lo=None,hi=None,qall=None)
    return out

# ---------------- ensemble + score ----------------
def ens_nn(runs,use_band):
    E={}
    for wi in runs[0]:
        s,e,_=runs[0][wi]; pe={}
        for d in DMAS:
            med=np.mean([r[wi][2][d]["median"] for r in runs],0)
            if use_band:
                lo=np.mean([r[wi][2][d]["lo"] for r in runs],0); hi=np.mean([r[wi][2][d]["hi"] for r in runs],0)
                qa=[np.mean([r[wi][2][d]["qall"][k] for r in runs],0) for k in range(len(QUANTILES))]
                pe[d]=dict(median=med,lo=lo,hi=hi,qall=qa)
            else: pe[d]=dict(median=med,lo=None,hi=None,qall=None)
        E[wi]=(s,e,pe)
    return E
def score(m,pred,prob):
    rows=[]
    for wi,(s,e,p) in pred.items():
        for d in DMAS:
            y=m[d].iloc[s:e].values.astype(float).copy()
            imp=m[d+"_imp"].iloc[s:e].values.astype(bool); y[imp]=np.nan; P=p[d]
            pi1,pi2,pi3=pi_metrics(y,P["median"]); row=dict(window=str(m.index[s].date()),dma=d,PI1=pi1,PI2=pi2,PI3=pi3)
            if prob and P["lo"] is not None:
                cov,mpiw,pb,wk=prob_metrics(y,P["lo"],P["hi"],P["qall"],QUANTILES)
                row.update(COV90=cov,MPIW=mpiw,Pinball=pb,Winkler=wk)
            rows.append(row)
    return pd.DataFrame(rows)

if __name__=="__main__":
    t0=time.time(); m=load_model(); wins=test_windows(m)
    refs={d:seasonal_ref(m[d].values.astype(np.float32)) for d in DMAS}
    print(f"Device:{DEVICE} | TEST official weeks | seeds={N_SEEDS} | CQR_COVER={CQR_COVER}")

    details={}  # method -> detail df
    # ---- point baselines ----
    details["Naive-4wk"]=score(m,{wi:(s,e,naive_pred(m,s)) for wi,(s,e) in enumerate(wins,1)},False)
    print("  Naive-4wk done")
    details["XGBoost"]=score(m,{wi:(s,e,xgb_pred(m,s)) for wi,(s,e) in enumerate(wins,1)},False)
    print("  XGBoost done")
    # ---- neural methods (ensembled) ----
    def run_nn(use_band,use_cons):
        runs=[]
        for si in range(N_SEEDS):
            seed=42+si; pw={}
            for wi,(s,e) in enumerate(wins,1):
                ds,ws=fit_stats(m,s); net=train_nn(m,s,ds,ws,refs,seed,use_band,use_cons)
                corr=cqr_corr(net,m,s,ds,ws,refs) if use_band else {d:0.0 for d in DMAS}
                pw[wi]=(s,e,predict_nn(net,m,s,ds,ws,corr,use_band))
            runs.append(pw)
        return ens_nn(runs,use_band)
    details["LSTM-point"]=score(m,run_nn(False,False),False); print("  LSTM-point done")
    details["CGPF-noCons"]=score(m,run_nn(True,False),True);  print("  CGPF-noCons done")
    details["CGPF"]=score(m,run_nn(True,True),True);          print("  CGPF done")

    # ---- main comparison table ----
    rows=[]
    for name,df in details.items():
        r=dict(Method=name,PI1=df.PI1.mean(),PI2=df.PI2.mean(),PI3=df.PI3.mean())
        if "COV90" in df: r.update(COV90=df.COV90.mean(),MPIW=df.MPIW.mean(),Pinball=df.Pinball.mean(),Winkler=df.Winkler.mean())
        rows.append(r)
    T=pd.DataFrame(rows).set_index("Method")
    print("\n=== PHASE B: official-week comparison (mean over 4 weeks x 10 DMAs) ===")
    print(T.round(4).to_string())
    T.to_csv(OUT_DIR+"phaseB_comparison.csv")
    details["CGPF"].groupby("dma")[["PI1","COV90","Winkler"]].mean().round(3).to_csv(OUT_DIR+"phaseB_cgpf_perzone.csv")

    # ---- B2: paired Wilcoxon (per-DMA-week) ----
    def paired(a,b,col):
        j=details[a].merge(details[b],on=["window","dma"],suffixes=("_a","_b"))
        x=j[col+"_a"].values; y=j[col+"_b"].values; ok=~(np.isnan(x)|np.isnan(y))
        if ok.sum()<5: return np.nan,np.nan
        try: stat,p=wilcoxon(x[ok],y[ok]); return float(p), float(np.mean(x[ok]-y[ok]))
        except: return np.nan,np.nan
    print("\n=== B2: paired Wilcoxon tests (per-DMA-week) ===")
    sig=[]
    for b in ["Naive-4wk","XGBoost","LSTM-point"]:
        p,md=paired("CGPF",b,"PI1"); sig.append(dict(test=f"CGPF vs {b}",metric="PI1",mean_diff=md,p_value=p))
        print(f"  CGPF vs {b:11s} PI1: mean diff={md:+.3f} (neg=CGPF better)  p={p:.4f}")
    p,md=paired("CGPF","CGPF-noCons","Winkler"); sig.append(dict(test="CGPF vs CGPF-noCons",metric="Winkler",mean_diff=md,p_value=p))
    print(f"  CGPF vs CGPF-noCons Winkler: mean diff={md:+.3f} (neg=constraints better)  p={p:.4f}")
    pd.DataFrame(sig).to_csv(OUT_DIR+"phaseB_significance.csv",index=False)
    print(f"\nSaved -> {OUT_DIR}phaseB_*.csv   ({time.time()-t0:.0f}s)")
