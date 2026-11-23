"""Earlier burden score; train and test periods share no survey waves."""
import pandas as pd
import numpy as np
from rebuild import usa,korea,coverage,med,out
def get_prior(q):
 h=q.groupby('hh');valid=coverage(q)&q.inc.gt(0)&q['size'].eq(h.expense.transform('count'))
 return (h.expense.transform('sum')/q.inc).where(valid)
def cutoff(score,weight):
 order=np.argsort(-score,kind='stable');s=score[order];w=weight[order];c=.15*w.sum()
 v=s[np.searchsorted(w.cumsum(),c)];return v,np.clip((c-w[s>v].sum())/w[s==v].sum(),0,1)
def score_result(wtrain,wtest,xtrain,xtest,y,win):
 # Each training window contributes the same total weight.
 wtrain=wtrain.copy()
 for j in np.unique(win):
  idx=win==j
  if wtrain[idx].sum()==0:return np.full(7,np.nan)
  wtrain[idx]/=wtrain[idx].sum()
 cut,tie=cutoff(xtrain,wtrain);selected=(xtest>cut)+tie*(xtest==cut)
 rate=np.average(selected,weights=wtest);risk=np.average(y,weights=wtest);yield_=np.average(selected*y,weights=wtest)
 bench,tieb=cutoff(xtest,wtest)
 selected_b=(xtest>bench)+tieb*(xtest==bench)
 actual=np.average(selected_b,weights=wtest)
 assert np.isclose(actual,.15)
 yield_b=np.average(selected_b*y,weights=wtest)
 return [cut,rate*100,yield_*1000,(yield_-risk*rate)*1000,actual*100,yield_b*1000,(yield_b-risk*actual)*1000]
def test(name):
 waves,_=(usa if name=='HRS' else korea)()
 frame=pd.read_pickle(out/(name+'_temp.pkl'));g=frame[frame.mid.eq(1)].copy()
 last=g.win.max();g=g[g.win.eq(last)|g.win.le(last-3)].copy()
 # Preserve the full, later validation population; exclude overlapping people and household origins from training.
 g['prior']=np.nan
 for win,part in g.groupby('win'):
  if 'expense' in waves[win]:g.loc[g.win.eq(win),'prior']=get_prior(waves[win]).reindex(part.index).to_numpy()
 g=g.dropna(subset=['prior'])
 if name=='HRS':g['family']=g.cluster
 else:
  import zipfile,io
  z=zipfile.ZipFile('klosa_work/STATA.zip')
  for win,part in g.groupby('win'):
   fl=next(n for n in z.namelist() if n.startswith(f'str{win:02}'))
   h=pd.read_stata(io.BytesIO(z.read(fl)),columns=['pid','hhid'],convert_categoricals=False).set_index('pid').hhid
   g.loc[g.win.eq(win),'family']=h.reindex(part.index).to_numpy()
 assert g.family.notna().all()
 late=g[g.win.eq(last)];early=g[g.win.lt(last)];train=early[~early.family.isin(late.family)&~early.id.isin(late.id)]
 assert train.win.max()+2 < last and set(train.family).isdisjoint(late.family) and set(train.id).isdisjoint(late.id)
 s=train.prior.to_numpy();x=late.prior.to_numpy();w=train.wt.to_numpy(float);v=late.wt.to_numpy(float);y=(late.oop>.1*late.den).to_numpy(float);window=train.win.to_numpy()
 point=score_result(w,v,s,x,y,window)
 rng=np.random.default_rng(20261122);reps=[]
 a,ia=np.unique(train.family,return_inverse=True);b,ib=np.unique(late.family,return_inverse=True)
 for j in range(1000):
  wa=w*rng.multinomial(len(a),np.full(len(a),1/len(a)))[ia]
  vb=v*rng.multinomial(len(b),np.full(len(b),1/len(b)))[ib]
  reps.append(score_result(wa,vb,s,x,y,window))
 ci=np.nanpercentile(reps,[2.5,97.5],axis=0)
 return [[name,str(sorted(set(map(int,window)))),int(last),len(early),len(train),len(late),int(y.sum()),metric,val,lo,hi] for metric,val,lo,hi in zip(['prior_ratio_cutoff','selected_percent','identified_per_1000','excess_vs_random_same_actual_capacity_per_1000','reranked_selected_percent','reranked_identified_per_1000','reranked_excess_vs_random_per_1000'],point,*ci)]
if __name__=='__main__':
 rows=test('KLoSA')+test('HRS');df=pd.DataFrame(rows,columns=['cohort','training_baseline_waves','test_baseline_wave','training_before_purge','training_after_purge','test_n','test_cases','metric','estimate','lower95','upper95']);df.to_csv(out/'validation.csv',index=False);print(df.round(2).to_string(index=False))
