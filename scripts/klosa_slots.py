"""KLoSA: participant bootstrap, pooled survey weights, capacity per window."""
import numpy as np
import pandas as pd
from rebuild import out
from window_check import panel_windows

def main():
    frame=pd.concat([g for _,g in panel_windows()])
    print('KLoSA eligible person-windows:',len(frame))
    labels=['spending','spending divided by income-group median','lower-income group first','contemporaneous actual ratio']
    units,group=np.unique(frame.id,return_inverse=True)
    parts=[]
    for _,idx in frame.groupby('win').indices.items():
        g=frame.iloc[idx]
        case=g.oop.gt(.1*g.den)
        flags=np.column_stack([case & g.low,case & ~g.low,~case])
        ranks=[np.unique(-score,return_inverse=True)[1] for score in [g.oop,g.oop/g.ref,g.low.astype(int),g.oop/g.den]]
        parts.append((group[idx],g.wt.to_numpy(),flags,ranks))
    def count(freq):
        total=np.zeros((4,3))
        denom=0
        for ids,wt,flags,ranks in parts:
            w=wt*freq[ids]
            denom+=w.sum()
            for j,rank in enumerate(ranks):
                mass=np.bincount(rank,weights=w)
                before=mass.cumsum()-mass
                take=np.clip((.15*w.sum()-before)/np.maximum(mass,1e-200),0,1)[rank]
                total[j]+=(w*take)@flags
        total*=1000/denom
        np.testing.assert_allclose(total.sum(axis=1),150)
        return np.column_stack([total,total[:,:2].sum(axis=1)])
    point=count(np.ones(len(units)))
    rng=np.random.default_rng(20261122)
    reps=np.stack([count(rng.multinomial(len(units),np.full(len(units),1/len(units)))) for _ in range(1500)])
    ci=np.percentile(reps-reps[:,:1],[2.5,97.5],axis=0)
    rows=[[rule,metric,point[j,k],point[j,k]-point[0,k],*ci[:,j,k]] for j,rule in enumerate(labels)
          for k,metric in enumerate(['lower-income cases','higher-income cases','below or at burden threshold','all cases'])]
    tab=pd.DataFrame(rows,columns=['rule','assessed_per_1000_by_group','estimate','change_vs_spending','lower95_change','upper95_change'])
    tab.to_csv(out/'klosa_slots.csv',index=False)
    print(tab.round(2).to_string(index=False))
if __name__=='__main__':
    main()
