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
res=[]
for imp,g in d.groupby('implicat'):
 assert g.index.is_unique
 print('imputation, n, newly limited, gap quartiles',imp,len(g),g.mid.sum(),g.gap.quantile([0,.25,.5,.75,1]).values)
 x=np.column_stack([np.ones(len(g)),g.low,g.mid,g.low*g.mid,g.age_int,g.gender.eq(2),pd.get_dummies(g.country,drop_first=True)]).astype(float)
 inv=np.linalg.inv(x.T@x)
 for th in [.05,.10,.25]:
  y=(g.oop>g.den*th).to_numpy(float)
  coef=inv@x.T@y
  sc=pd.DataFrame(x*(y-x@coef)[:,None]).groupby(g.hhid7.to_numpy()).sum().to_numpy()
  n,k=x.shape;G=len(sc);cov=inv@sc.T@sc@inv*(G/(G-1))*((n-1)/(n-k))
  high=x.mean(axis=0);high[1:4]=[0,1,0];low=high.copy();low[1:4]=[1,1,1]
  inter=np.eye(k)[3]
  for label,L in [('lower',low),('higher',high),('gap',low-high),('interaction',inter)]:
   res.append([imp,th,label,100*(L@coef),10000*(L@cov@L)])
results=[]
for (th,metric),g in pd.DataFrame(res,columns=['imp','threshold','metric','q','u']).groupby(['threshold','metric']):
 q=g.q.mean();w=g.u.mean();b=1.2*g.q.var();se=np.sqrt(w+b);crit=t.ppf(.975,4*(1+w/b)**2) if b>0 else 1.96
 results.append([th,metric,q,q-crit*se,q+crit*se])
r=pd.DataFrame(results,columns=['income_share_threshold','estimate','percent_or_pp','lower95','upper95']);print(r.to_string(index=False));r.to_csv('rerun/SHARE_estimates.csv',index=False)
# Paired retrospective comparison at fixed capacity; income-only uses the same binary group available to both rules.
from scipy.stats import norm
records=[];rng=np.random.default_rng(20261122)
for imp,part in d[d.mid.eq(True)].groupby('implicat'):
 totals=np.zeros((4,2001,3)); denominators=np.zeros((2001,3))
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
 rates=100*totals/denominators;rates[:,:,:2]=100-rates[:,:,:2]
 records.append(rates)
x=np.array(records)
summary=[]
for j,label in enumerate(['spending','income-informed','income group only','observed ratio benchmark']):
 for typ,data in [('rate',x[:,j]),('change vs spending',x[:,j]-x[:,0])]:
  q=data[:,0,:];w=data[:,1:,:].var(axis=1,ddof=1).mean(axis=0);b=1.2*q.var(axis=0,ddof=1);se=np.sqrt(w+b)
  crit=np.full(3,norm.ppf(.975));ix=b>0;crit[ix]=t.ppf(.975,4*(1+w[ix]/b[ix])**2)
  for k,metric in enumerate(['lower-income miss','higher-income miss','overall identification']):
   mean=q[:,k].mean();summary.append([label,typ,metric,mean,mean-crit[k]*se[k],mean+crit[k]*se[k]])
scr=pd.DataFrame(summary,columns=['rule','quantity','metric','estimate','lower95','upper95']);scr.to_csv('rerun/SHARE_allocation.csv',index=False)
print(scr[scr.quantity.eq('rate')].round(1).to_string(index=False))
