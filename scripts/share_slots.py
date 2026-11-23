"""Five imputations, with equal country contributions to each per-thousand estimate."""
from pathlib import Path
import glob,zipfile,io
import numpy as np,pandas as pd
from scipy.stats import t
def read(w,mod,cols):
 with zipfile.ZipFile(f'sharework/sharew{w}_rel9-0-0_ALL_datasets_stata.zip') as z:
  n=next(n for n in z.namelist() if n.endswith('_'+mod+'.dta'))
  return pd.read_stata(io.BytesIO(z.read(n)),columns=['mergeid']+cols,convert_categoricals=False).set_index('mergeid')
def adl(w):
 g=read(w,'ph',[f'ph049d{k}' for k in [1,3,4,5,6]])
 return g.eq(1).any(axis=1).where(g.isin([0,1]).all(axis=1))
a=read(5,'cv_r',['hhid5','hhsize','age_int','country','gender'])
a['base']=adl(5);a['mid']=adl(6)
def month(w):
 b=read(w,'cv_r',['int_year','int_month'])
 return (b.int_year*12+b.int_month).where(b.int_month.between(1,12)&b.int_year.gt(2000))
b=read(7,'cv_r',['hhid7','hhsize']);b['gap']=month(7)-month(6)
a=a.join(b[['hhid7','gap']]);a=a[a.age_int.between(60,120)&a.base.eq(False)&a.hhsize.gt(0)&a.hhid5.ne('')]
i=read(5,'gv_imputations',['implicat','thinc']).join(a,how='inner')
i=i[i.thinc.gt(0)].copy();i['inc']=i.thinc/i.hhsize.pow(.5)
i['low']=i.inc.le(i.groupby(['country','implicat']).inc.transform('median'))
i['ref']=i.groupby(['country','implicat','low']).thinc.transform('median')
cols=['inpat7','outpa7','drugs7','nurs7','aapt7']
e=read(7,'gv_imputations',['implicat','thinc']+cols).join(b)
e=e[e.hhid7.ne('')&e.hhsize.gt(0)].copy();e['ok']=e[cols].ge(0).all(axis=1);e['oop']=e[cols].sum(axis=1)
h=e.groupby(['hhid7','implicat']).agg(n=('hhsize','size'),lo=('hhsize','min'),hi=('hhsize','max'),ok=('ok','all'),den=('thinc','min'),mx=('thinc','max'),oop=('oop','sum'))
h=h[h.n.eq(h.lo)&h.lo.eq(h.hi)&h.ok&h.den.gt(0)&h.den.eq(h.mx)]
d=i.join(h,on=['hhid7','implicat'],how='inner');d=d[d.mid.notna()&d.gap.ge(13)&d.gender.isin([1,2])].copy()
from scipy.stats import norm
records=[];rng=np.random.default_rng(20261122)
for imp,part in d[d.mid.eq(True)].groupby('implicat'):
 totals=np.zeros((4,2001,3)); denominators=np.zeros((2001,3))
 countries=part.country.nunique()
 assert countries==10
 print("SHARE",imp,len(part),countries)
 for _,g in part.groupby('country'):
  hh,ix=np.unique(g.hhid7,return_inverse=True);mult=np.vstack([np.ones(len(hh)),rng.multinomial(len(hh),np.full(len(hh),1/len(hh)),2000)])[:,ix]
  size=mult.sum(axis=1,keepdims=True);wt=mult/size
  high=g.oop.gt(.1*g.den);mask=np.column_stack([high&g.low,high&~g.low,high])
  denominators+=wt@mask
  for j,score in enumerate([g.oop,g.oop/g.ref,g.low.astype(int),g.oop/g.den]):
   lev,idx=np.unique(-score.to_numpy(),return_inverse=True)
   count=mult@np.eye(len(lev))[idx];before=count.cumsum(axis=1)-count
   select=np.clip((.15*size-before)/np.maximum(count,1),0,1)[:,idx]
   assert np.allclose((wt*select).sum(axis=1),.15)
   totals[j]+=(wt*select)@mask
 slots=1000*totals/countries
 assert np.all(slots[:,:,-1]<=150+1e-7)
 records.append(slots)
x=np.asarray(records)
labels=['spending','spending divided by income-group median','lower-income group first','contemporaneous actual ratio']
metrics=['lower-income cases','higher-income cases','all cases']
rows=[]
for j,rule in enumerate(labels):
 diff=x[:,j]-x[:,0]
 point=x[:,j,0].mean(axis=0)
 q=diff[:,0]
 within=diff[:,1:].var(axis=1,ddof=1).mean(axis=0)
 between=1.2*q.var(axis=0,ddof=1)
 critical=np.full(3,norm.ppf(.975))
 positive=between>0
 critical[positive]=t.ppf(.975,4*(1+within[positive]/between[positive])**2)
 change=q.mean(axis=0)
 margin=critical*np.sqrt(within+between)
 for k,metric in enumerate(metrics):
  rows.append([rule,metric,point[k],change[k],change[k]-margin[k],change[k]+margin[k]])
tab=pd.DataFrame(rows,columns=['rule','assessed_per_1000_by_group','estimate','change_vs_spending','lower95_change','upper95_change'])
tab.to_csv('review/share_slots.csv',index=False)
print(tab.round(2).to_string(index=False))
