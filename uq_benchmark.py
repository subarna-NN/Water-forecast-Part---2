"""
=====================================================================
 PHASE C - FAIR uncertainty-method benchmark (fixed pipeline, TEST once)
=====================================================================
"""
import os, warnings, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

DATA_PATH="/content/model_ready_v2.csv"; OUT_DIR="/content/"
N_SEEDS=3
LOOKBACK,HORIZON,STRIDE=168,168,6
HIDDEN,EMB_DIM,LAYERS=64,8,1
EPOCHS,PATIENCE,BATCH,LR,VAL_FRAC=40,6,128,1e-3,0.15
CAL_SPAN_W=8
LAM_Q,LAM_VOL,LAM_SMOOTH=0.3,0.02,0.005
QUANTILES=[0.05,0.25,0.50,0.75,0.95]; Q_LO,Q_MED,Q_HI=0,2,4
ZQ=[-1.6449,-0.6745,0.0,0.6745,1.6449]
ALPHA,CQR_COVER=0.10,0.91
MC_T,MC_P,AR_SAMPLES=30,0.20,100
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DMAS=[f"DMA {c}" for c in "ABCDEFGHIJ"]
WEATHER=["Rain","Temperature","Humidity","Windspeed"]
CTX=WEATHER+["hour_sin","hour_cos","wk_sin","wk_cos","holiday"]; NCTX=len(CTX)

def load():
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"{DATA_PATH} not found. Upload model_ready_v2.csv.")
    m=pd.read_csv(DATA_PATH,index_col=0); m.index=pd.to_datetime(m.index,utc=True); return m
def test_windows(m):
    ev=m["eval_week"].values; b=[]; s=None
    for i,v in enumerate(ev):
        if v and s is None: s=i
        elif not v and s is not None: b.append((s,i)); s=None
    if s is not None: b.append((s,len(ev)))
    return b
def pi1_metric(y,yh):
    e=np.abs(np.asarray(y)-np.asarray(yh)); d=e[:24][~np.isnan(e[:24])]; return np.mean(d) if len(d) else np.nan
def prob_metrics(y,lo,hi,qall,taus):
    y=np.asarray(y,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float); ok=~np.isnan(y)
    if ok.sum()==0: return (np.nan,)*4
    cov=np.mean((y[ok]>=lo[ok])&(y[ok]<=hi[ok])); mpiw=np.mean(hi[ok]-lo[ok])
    pb=[np.mean(np.maximum(t*(y[ok]-qall[k][ok]),(t-1)*(y[ok]-qall[k][ok]))) for k,t in enumerate(taus)]
    w=(hi[ok]-lo[ok]).copy(); b=y[ok]<lo[ok]; a=y[ok]>hi[ok]
    w[b]+=2/ALPHA*(lo[ok][b]-y[ok][b]); w[a]+=2/ALPHA*(y[ok][a]-hi[ok][a])
    return cov,mpiw,float(np.mean(pb)),float(np.mean(w))
def fit_stats(m,origin):
    cal=origin-CAL_SPAN_W*168
    d={x:(float(np.nanmean(m[x].values[:cal])),float(np.nanstd(m[x].values[:cal])+1e-6)) for x in DMAS}
    w={x:(float(np.nanmean(m[x].values[:cal])),float(np.nanstd(m[x].values[:cal])+1e-6)) for x in WEATHER}
    return d,w
def scaled_ctx(m,ws):
    base=m[CTX].values.astype(np.float32).copy()
    for j,w in enumerate(WEATHER): mu,sd=ws[w]; base[:,j]=(base[:,j]-mu)/sd
    return base
def build(m,lo,hi,ds,ws):
    base=scaled_ctx(m,ws); E,D,T,I=[],[],[],[]
    for di,d in enumerate(DMAS):
        mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
        first=max(0,lo-LOOKBACK); last=hi-HORIZON-LOOKBACK
        for s in range(first,last+1,STRIDE):
            ee=s+LOOKBACK; de=ee+HORIZON
            if ee<lo: continue
            E.append(np.column_stack([dem[s:ee],base[s:ee]])); D.append(base[ee:de]); T.append(dem[ee:de]); I.append(di)
    return np.asarray(E,np.float32),np.asarray(D,np.float32),np.asarray(T,np.float32),np.asarray(I,np.int64)
def within_dma_split(I):
    tr=np.zeros(len(I),bool)
    for di in range(len(DMAS)):
        pos=np.where(I==di)[0]; cut=int(len(pos)*(1-VAL_FRAC)); tr[pos[:cut]]=True
    return tr

# ---------------- models ----------------
class QNet(nn.Module):
    def __init__(s,ne,nd,nq):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True); s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.mu=nn.Linear(HIDDEN,1); s.dn=nn.Linear(HIDDEN,nq//2); s.up=nn.Linear(HIDDEN,nq//2)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        mu=s.mu(o); low=torch.flip(mu-torch.cumsum(F.softplus(s.dn(o)),-1),[-1]); high=mu+torch.cumsum(F.softplus(s.up(o)),-1)
        return torch.cat([low,mu,high],-1),mu
class GNet(nn.Module):
    def __init__(s,ne,nd):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True); s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.mu=nn.Linear(HIDDEN,1); s.ls=nn.Linear(HIDDEN,1)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        return s.mu(o).squeeze(-1),s.ls(o).squeeze(-1)
class MNet(nn.Module):
    def __init__(s,ne,nd):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.enc=nn.LSTM(ne+EMB_DIM,HIDDEN,LAYERS,batch_first=True); s.dec=nn.LSTM(nd+EMB_DIM,HIDDEN,LAYERS,batch_first=True)
        s.do=nn.Dropout(MC_P); s.head=nn.Linear(HIDDEN,1)
    def forward(s,ex,dx,did):
        e=s.emb(did); ee=e.unsqueeze(1).expand(-1,ex.size(1),-1); ed=e.unsqueeze(1).expand(-1,dx.size(1),-1)
        _,(h,c)=s.enc(torch.cat([ex,ee],-1)); o,_=s.dec(torch.cat([dx,ed],-1),(h,c))
        return s.head(s.do(o)).squeeze(-1)
class ARNet(nn.Module):
    def __init__(s,ncov):
        super().__init__(); s.emb=nn.Embedding(len(DMAS),EMB_DIM)
        s.lstm=nn.LSTM(1+ncov+EMB_DIM,HIDDEN,LAYERS,batch_first=True); s.mu=nn.Linear(HIDDEN,1); s.ls=nn.Linear(HIDDEN,1)
    def seq(s,y,cov,did):
        e=s.emb(did).unsqueeze(1).expand(-1,cov.size(1),-1); x=torch.cat([y.unsqueeze(-1),cov,e],-1); o,hc=s.lstm(x); return o,hc
    def step(s,prev,cov,did,hc):
        e=s.emb(did).unsqueeze(1); x=torch.cat([prev.view(-1,1,1),cov.unsqueeze(1),e],-1); o,hc=s.lstm(x,hc)
        return s.mu(o).squeeze(1).squeeze(-1),s.ls(o).squeeze(1).squeeze(-1),hc
def nll(mu,ls,y): s=F.softplus(ls)+1e-3; return torch.mean(0.5*np.log(2*np.pi)+torch.log(s)+0.5*((y-mu)/s)**2)

# ---------------- training ----------------
def _loaders(E,D,T,I):
    trm=within_dma_split(I)
    def dl(sel,sh): return DataLoader(TensorDataset(torch.tensor(E[sel]),torch.tensor(D[sel]),torch.tensor(T[sel]),torch.tensor(I[sel])),batch_size=BATCH,shuffle=sh)
    return dl(trm,True),dl(~trm,False)
def _es_train(net,tr,va,lossf,vlossf):
    opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0
    for ep in range(EPOCHS):
        net.train()
        for b in tr:
            b=[x.to(DEVICE) for x in b]; opt.zero_grad(); lossf(net,b).backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for b in va:
                b=[x.to(DEVICE) for x in b]; vs+=vlossf(net,b)*len(b[0]); vn+=len(b[0])
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net
def train_quant(m,o,ds,ws,seed,cons):
    torch.manual_seed(seed); np.random.seed(seed); E,D,T,I=build(m,0,o-CAL_SPAN_W*168,ds,ws); tr,va=_loaders(E,D,T,I)
    net=QNet(1+NCTX,NCTX,len(QUANTILES)-1).to(DEVICE)
    def lf(n,b):
        eb,db,tb,ib=b; q,mu=n(eb,db,ib); L=F.l1_loss(q[...,Q_MED],tb)
        for k,t in enumerate(QUANTILES):
            if k==Q_MED: continue
            qb=mu.detach()+(q-mu); d=tb-qb[...,k]; L=L+LAM_Q*torch.mean(torch.maximum(t*d,(t-1)*d))
        if cons:
            md=q[...,Q_MED]; L=L+LAM_VOL*torch.mean((md.mean(1))**2)+LAM_SMOOTH*torch.mean((md[:,2:]-2*md[:,1:-1]+md[:,:-2])**2)
        return L
    def vf(n,b): eb,db,tb,ib=b; q,_=n(eb,db,ib); return F.l1_loss(q[...,Q_MED],tb).item()
    return _es_train(net,tr,va,lf,vf)
def train_gauss(m,o,ds,ws,seed):
    torch.manual_seed(seed); np.random.seed(seed); E,D,T,I=build(m,0,o-CAL_SPAN_W*168,ds,ws); tr,va=_loaders(E,D,T,I)
    net=GNet(1+NCTX,NCTX).to(DEVICE)
    def lf(n,b): eb,db,tb,ib=b; mu,ls=n(eb,db,ib); return nll(mu,ls,tb)
    def vf(n,b): eb,db,tb,ib=b; mu,ls=n(eb,db,ib); return nll(mu,ls,tb).item()
    return _es_train(net,tr,va,lf,vf)
def train_mc(m,o,ds,ws,seed):
    torch.manual_seed(seed); np.random.seed(seed); E,D,T,I=build(m,0,o-CAL_SPAN_W*168,ds,ws); tr,va=_loaders(E,D,T,I)
    net=MNet(1+NCTX,NCTX).to(DEVICE)
    def lf(n,b): eb,db,tb,ib=b; return F.mse_loss(n(eb,db,ib),tb)
    def vf(n,b): eb,db,tb,ib=b; return F.mse_loss(n(eb,db,ib),tb).item()
    return _es_train(net,tr,va,lf,vf)
def train_ar(m,o,ds,ws,seed):
    torch.manual_seed(seed); np.random.seed(seed); base=scaled_ctx(m,ws); L=LOOKBACK+HORIZON; Y,C,I=[],[],[]
    for di,d in enumerate(DMAS):
        mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
        for s in range(0,o-L+1,STRIDE): Y.append(dem[s:s+L]); C.append(base[s:s+L]); I.append(di)
    Y=np.asarray(Y,np.float32); C=np.asarray(C,np.float32); I=np.asarray(I,np.int64); n=len(Y); cut=int(n*(1-VAL_FRAC))
    def dl(a,b,sh): return DataLoader(TensorDataset(torch.tensor(Y[a:b]),torch.tensor(C[a:b]),torch.tensor(I[a:b])),batch_size=BATCH,shuffle=sh)
    tr,va=dl(0,cut,True),dl(cut,n,False); net=ARNet(NCTX).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=LR); best,bs,bad=np.inf,None,0   # AR custom loop (seq outputs)
    for ep in range(EPOCHS):
        net.train()
        for yb,cb,ib in tr:
            yb,cb,ib=yb.to(DEVICE),cb.to(DEVICE),ib.to(DEVICE); opt.zero_grad()
            o_,_=net.seq(yb[:,:-1],cb[:,1:],ib); mu=net.mu(o_).squeeze(-1); ls=net.ls(o_).squeeze(-1)
            loss=nll(mu,ls,yb[:,1:]); loss.backward(); opt.step()
        net.eval(); vs=vn=0
        with torch.no_grad():
            for yb,cb,ib in va:
                yb,cb,ib=yb.to(DEVICE),cb.to(DEVICE),ib.to(DEVICE)
                o_,_=net.seq(yb[:,:-1],cb[:,1:],ib); mu=net.mu(o_).squeeze(-1); ls=net.ls(o_).squeeze(-1)
                vs+=nll(mu,ls,yb[:,1:]).item()*len(yb); vn+=len(yb)
        v=vs/max(vn,1)
        if v<best-1e-5: best,bs,bad=v,{k:val.cpu().clone() for k,val in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE: break
    if bs: net.load_state_dict(bs)
    return net

# ---------------- RAW predict (one origin -> dict dma->(median,lo,hi,qall)) ----------------
def raw_quant(net,m,o,ds,ws):
    base=scaled_ctx(m,ws); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[o-LOOKBACK:o],base[o-LOOKBACK:o]])[None]; dec=base[o:o+HORIZON][None]
            q,_=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]).to(DEVICE))
            q=q.cpu().numpy()[0]*sd+mu; qa=[np.clip(q[:,k],0,None) for k in range(len(QUANTILES))]
            out[d]=dict(median=qa[Q_MED],lo=np.clip(q[:,Q_LO],0,None),hi=np.clip(q[:,Q_HI],0,None),qall=qa)
    return out
def raw_gauss(net,m,o,ds,ws):
    base=scaled_ctx(m,ws); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.column_stack([dem[o-LOOKBACK:o],base[o-LOOKBACK:o]])[None]; dec=base[o:o+HORIZON][None]
            mn,ls=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]).to(DEVICE))
            mn=mn.cpu().numpy()[0]; sg=F.softplus(ls).cpu().numpy()[0]+1e-3
            qa=[np.clip((mn+z*sg)*sd+mu,0,None) for z in ZQ]
            out[d]=dict(median=qa[Q_MED],lo=qa[Q_LO],hi=qa[Q_HI],qall=qa)
    return out
def raw_mc(net,m,o,ds,ws):
    base=scaled_ctx(m,ws); out={}; net.train()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            enc=np.repeat(np.column_stack([dem[o-LOOKBACK:o],base[o-LOOKBACK:o]])[None],MC_T,0)
            dec=np.repeat(base[o:o+HORIZON][None],MC_T,0)
            s=net(torch.tensor(enc).to(DEVICE),torch.tensor(dec).to(DEVICE),torch.tensor([di]*MC_T).to(DEVICE)).cpu().numpy()*sd+mu
            qa=[np.clip(np.quantile(s,t,axis=0),0,None) for t in QUANTILES]
            out[d]=dict(median=qa[Q_MED],lo=qa[Q_LO],hi=qa[Q_HI],qall=qa)
    return out
def raw_ar(net,m,o,ds,ws):
    base=scaled_ctx(m,ws); out={}; net.eval()
    with torch.no_grad():
        for di,d in enumerate(DMAS):
            mu,sd=ds[d]; dem=(m[d].values.astype(np.float32)-mu)/sd
            cov=base[o-LOOKBACK:o+HORIZON]                      # local idx0 = o-LOOKBACK
            yh=dem[o-LOOKBACK:o]                                # y_{o-L}..y_{o-1}
            did=torch.tensor([di]*AR_SAMPLES).to(DEVICE)
            # FIX: condition on steps 1..L-1 (prev=y_{o-L..o-2}, cov=cov_{o-L+1..o-1}); cov_o used ONLY in prediction
            prev_in=torch.tensor(np.tile(yh[:LOOKBACK-1],(AR_SAMPLES,1))).to(DEVICE)
            cov_in=torch.tensor(np.tile(cov[1:LOOKBACK],(AR_SAMPLES,1,1))).to(DEVICE)
            _,hc=net.seq(prev_in,cov_in,did)
            prev=torch.tensor(np.tile(yh[LOOKBACK-1],(AR_SAMPLES,))).float().to(DEVICE)   # y_{o-1}
            paths=[]
            for t in range(HORIZON):
                ct=torch.tensor(np.tile(cov[LOOKBACK+t],(AR_SAMPLES,1))).to(DEVICE)       # cov_o .. cov_{o+H-1}
                mn,ls,hc=net.step(prev,ct,did,hc); sg=F.softplus(ls)+1e-3
                prev=mn+sg*torch.randn_like(sg); paths.append(prev.cpu().numpy())
            s=np.stack(paths,1)*sd+mu
            qa=[np.clip(np.quantile(s,t,axis=0),0,None) for t in QUANTILES]
            out[d]=dict(median=qa[Q_MED],lo=qa[Q_LO],hi=qa[Q_HI],qall=qa)
    return out

# ---------------- ensemble raw + uniform conformal ----------------
def ens_raw(raw_fns,o):   # average raw predictions across seed-nets
    outs=[f(o) for f in raw_fns]; E={}
    for d in DMAS:
        E[d]=dict(median=np.mean([x[d]["median"] for x in outs],0),
                  lo=np.mean([x[d]["lo"] for x in outs],0),
                  hi=np.mean([x[d]["hi"] for x in outs],0),
                  qall=[np.mean([x[d]["qall"][k] for x in outs],0) for k in range(len(QUANTILES))])
    return E
def conformal_Q(ens_fn,m,test_o):
    scores={d:[] for d in DMAS}
    for co in range(test_o-CAL_SPAN_W*168, test_o-HORIZON+1, 168):
        if co-LOOKBACK<0: continue
        P=ens_fn(co)
        for d in DMAS:
            y=m[d].values[co:co+HORIZON].astype(float).copy(); imp=m[d+"_imp"].values[co:co+HORIZON].astype(bool); y[imp]=np.nan
            e=np.maximum(P[d]["lo"]-y, y-P[d]["hi"]); scores[d].append(e[~np.isnan(e)])
    Q={}
    for d in DMAS:
        a=np.concatenate(scores[d]) if scores[d] else np.array([0.0]); nE=len(a)
        Q[d]=float(np.quantile(a,min(1.0,np.ceil((nE+1)*CQR_COVER)/nE)))
    return Q

def run_method(m,wins,trainers,raw_fn,conform):
    """trainers: list of trained nets per window-seed via closure. Returns detail df."""
    rows=[]
    for wi,(s,e) in enumerate(wins,1):
        ds,ws=fit_stats(m,s)
        nets=[trainers(m,s,ds,ws,42+si) for si in range(N_SEEDS)]
        raw_fns=[(lambda oo,nn=nn: raw_fn(nn,m,oo,ds,ws)) for nn in nets]
        ensf=lambda oo: ens_raw(raw_fns,oo)
        Q=conformal_Q(ensf,m,s) if conform else {d:0.0 for d in DMAS}
        P=ensf(s)
        for d in DMAS:
            y=m[d].iloc[s:e].values.astype(float).copy(); imp=m[d+"_imp"].iloc[s:e].values.astype(bool); y[imp]=np.nan
            lo=np.clip(P[d]["lo"]-Q[d],0,None); hi=np.clip(P[d]["hi"]+Q[d],0,None)
            pi1=pi1_metric(y,P[d]["median"]); cov,mpiw,pb,wk=prob_metrics(y,lo,hi,P[d]["qall"],QUANTILES)
            rows.append(dict(window=str(m.index[s].date()),dma=d,PI1=pi1,COV90=cov,MPIW=mpiw,Pinball=pb,Winkler=wk))
    return pd.DataFrame(rows)

if __name__=="__main__":
    t0=time.time(); m=load(); wins=test_windows(m)
    print(f"Device:{DEVICE} | TEST weeks | seeds={N_SEEDS} | CQR_COVER={CQR_COVER}")
    SPECS={
      "QR":        (lambda m,o,ds,ws,sd: train_quant(m,o,ds,ws,sd,False), raw_quant, False),
      "CQR":       (lambda m,o,ds,ws,sd: train_quant(m,o,ds,ws,sd,False), raw_quant, True),
      "Gaussian":  (train_gauss, raw_gauss, True),
      "MC-Dropout":(train_mc,    raw_mc,    True),
      "DeepAR":    (train_ar,    raw_ar,    True),
      "CGPF":      (lambda m,o,ds,ws,sd: train_quant(m,o,ds,ws,sd,True),  raw_quant, True),
    }
    details={}
    for name,(tr,rf,cf) in SPECS.items():
        details[name]=run_method(m,wins,tr,rf,cf)
        ss=details[name][["PI1","COV90","Winkler"]].mean()
        print(f"  {name:11s} PI1={ss.PI1:.3f} COV90={ss.COV90:.3f} Winkler={ss.Winkler:.3f}   [{time.time()-t0:.0f}s]")

    T=pd.DataFrame({n:df[["PI1","COV90","MPIW","Pinball","Winkler"]].mean() for n,df in details.items()}).T
    print("\n=== PHASE C: fair UQ benchmark (all conformalised, fixed pipeline) ===")
    print(T.round(4).to_string()); T.to_csv(OUT_DIR+"phaseC_fair_uq.csv")
    # significance: CGPF vs each on Winkler
    print("\n=== Wilcoxon (CGPF vs each, Winkler, per-DMA-week) ===")
    sig=[]
    for b in [x for x in SPECS if x!="CGPF"]:
        j=details["CGPF"].merge(details[b],on=["window","dma"],suffixes=("_c","_b"))
        x=j.Winkler_c.values; y=j.Winkler_b.values; ok=~(np.isnan(x)|np.isnan(y))
        try: p=wilcoxon(x[ok],y[ok]).pvalue
        except: p=np.nan
        md=np.mean(x[ok]-y[ok]); sig.append(dict(vs=b,mean_diff=md,p=p))
        print(f"  CGPF vs {b:11s} Winkler diff={md:+.3f} (neg=CGPF better)  p={p:.4f}")
    pd.DataFrame(sig).to_csv(OUT_DIR+"phaseC_significance.csv",index=False)
    print(f"\nSaved -> {OUT_DIR}phaseC_*.csv   ({time.time()-t0:.0f}s)")
