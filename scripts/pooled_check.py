"""Pooled survey-weighted miss rates under a separate 15% allocation in each window."""
import numpy as np
import pandas as pd
from window_check import panel_windows
def summarize(g, cluster):
    units, groups=np.unique(g[cluster],return_inverse=True)
    pieces=[]
    for _, idx in g.groupby('win').indices.items():
        p=g.iloc[idx]
        high=p.oop.gt(.10*p.den)
        flags=np.column_stack([p.low & high,~p.low & high,high])
        ranks=[np.unique(-s,return_inverse=True)[1] for s in [p.oop,p.oop/p.ref]]
        pieces.append((groups[idx],p.wt.to_numpy(),flags,ranks))
    def estimate(freq):
        cases=np.zeros(3)
        found=np.zeros((2,3))
        for groups,weights,flags,ranks in pieces:
            w=weights*freq[groups]
            cases+=w@flags
            for i,rank in enumerate(ranks):
                mass=np.bincount(rank,weights=w)
                prior=mass.cumsum()-mass
                selected=np.clip((.15*w.sum()-prior)/np.maximum(mass,1e-200),0,1)[rank]
                assert np.isclose(w@selected,.15*w.sum())
                found[i]+=(w*selected)@flags
        rate=np.divide(100*found,cases,out=np.full_like(found,np.nan),where=cases>0)
        rate[:,:2]=100-rate[:,:2]
        return rate
    r=estimate(np.ones(len(units)))
    rng=np.random.default_rng(20261122)
    reps=[]
    for _ in range(600):
        b=estimate(rng.multinomial(len(units),np.full(len(units),1/len(units))))
        reps.append(b[1]-b[0])
    change=r[1]-r[0]
    se=np.nanstd(reps,axis=0,ddof=1)
    return pd.DataFrame(dict(metric=['lower miss','higher miss','overall identification'],spending=r[0],income_informed=r[1],change=change,normal_lower=change-1.96*se,normal_upper=change+1.96*se,percentile_lower=np.nanpercentile(reps,2.5,axis=0),percentile_upper=np.nanpercentile(reps,97.5,axis=0)))

if __name__ == '__main__':
    k=pd.concat([g for _,g in panel_windows()])
    h=pd.read_pickle('review/HRS_temp.pkl')
    h=h[h.mid.eq(1)&h.win.ge(13)]
    results=pd.concat([summarize(g,cluster).assign(cohort=name) for name,g,cluster in [('KLoSA',k,'id'),('HRS',h,'hh')]])
    results.to_csv('review/pooled_results.csv',index=False)
    print(results.round(2).to_string(index=False))
