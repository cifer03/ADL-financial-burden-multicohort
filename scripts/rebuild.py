"""Independent specification for auditing archived model counts. Run from project root."""
from pathlib import Path
import zipfile,io
import pandas as pd
import numpy as np
out=Path('review')
def month(y,m):
 y=pd.to_numeric(y,errors='coerce');m=pd.to_numeric(m,errors='coerce')
 return (y*12+m).where(y.between(1990,2030)&m.between(1,12))
def korea():
 z=zipfile.ZipFile('klosa_work/STATA.zip');waves={}
 for w in range(1,11):
  p=f'w{w:02}';r=pd.io.stata.StataReader(io.BytesIO(z.read(next(n for n in z.namelist() if n.startswith(f'str{w:02}')))),convert_categoricals=False)
  lab=r.variable_labels();hid=next((k for k in lab if k.startswith('HHID')),'hhid');wt='wgt_ac' if p+'wgt_ac' in lab else 'wgt_c'
  adl=['C201','C203','C204','C205','C206'];vis=['C330','C333','C334','C337','C340'];fields=['gender1','A002_age','hhsize','hhinc',wt,'oopm1','oopm2','C318','mniw_y','mniw_m']+vis+adl
  q=r.read(columns=['pid',hid]+[p+k for k in fields]).rename(columns={p+k:k for k in fields}).set_index('pid').rename(columns={hid:'hh',wt:'wt','A002_age':'age','hhsize':'size','hhinc':'inc'})
  q['female']=q.gender1.eq(5);q['adl']=q[adl].isin([3,5]).any(axis=1).where(q[adl].isin([1,3,5]).all(axis=1));q['date']=month(q.mniw_y,q.mniw_m)
  a=q.oopm1.where(q.oopm1.ge(0)).mask(q.oopm1.isna()&q.C318.eq(0),0)
  b=q.oopm2.where(q.oopm2.ge(0)).mask(q.oopm2.isna()&q[vis].eq(0).all(axis=1),0)
  q['expense']=(a+b)/2;waves[w]=q
 return waves,range(1,9)
def usa():
 z=zipfile.ZipFile('check/randhrs1992_2022v2_STATA.zip');path=Path('review/randhrs1992_2022v2.dta')
 if not path.exists():z.extract(path.name,path.parent)
 r=pd.io.stata.StataReader(path);fields=r.variable_labels();adl=['dressa','batha','eata','beda','toilta']
 cols=['hhidpn','hhid','rasex']+[f'{p}{w}{k}' for w in range(8,17) for p,ks in [('r',adl+['agey_e','wtresp','iwendy','iwendm','liwrefm','oopmdo']),('h',['hhres','hhid','itot'])] for k in ks if f'{p}{w}{k}' in fields]
 raw=pd.concat(pd.read_stata(path,columns=cols,chunksize=800,convert_categoricals=False)).set_index('hhidpn');waves={}
 for w in range(8,17):
  def get(p,k):return raw[f'{p}{w}{k}']
  q=pd.DataFrame(dict(cluster=raw.hhid,hh=get('h','hhid'),age=get('r','agey_e'),wt=get('r','wtresp'),inc=get('h','itot'),size=get('h','hhres'),female=raw.rasex.eq(2)))
  q['date']=month(get('r','iwendy'),get('r','iwendm'));items=raw[[f'r{w}{k}' for k in adl]]
  q['adl']=items.eq(1).any(axis=1).where(items.isin([0,1]).all(axis=1))
  if w>=10:
   ref=get('r','liwrefm').where(get('r','liwrefm').gt(0));exp=get('r','oopmdo')
   q['expense']=12*exp.where(exp.ge(0))/ref;q['start']=q.date-ref
  waves[w]=q
 return waves,range(8,15)
def coverage(q):
 g=q.groupby('hh');return q['size'].eq(g['size'].transform('size'))&q['size'].eq(g['size'].transform('min'))&q['size'].eq(g['size'].transform('max'))&q.inc.eq(g.inc.transform('max'))&q.inc.eq(g.inc.transform('min'))
def med(v,w):
 s=v.sort_values(kind='stable');weight=w.reindex(s.index);return s.iloc[np.searchsorted(weight.cumsum(),weight.sum()/2)]
def build(name,waves,windows):
 rs=[];fs=[]
 for w in windows:
  b,m,e=[waves[k] for k in [w,w+1,w+2]]
  keep=b.age.between(60,120)&b.adl.eq(False)&b.inc.gt(0)&b['size'].gt(0)&b.wt.gt(0)
  if name=='HRS':keep &=coverage(b)
  x=b.loc[keep].copy()
  eq=x.inc/np.sqrt(x['size']);x['low']=eq.le(med(eq,x.wt));x['ref']=x.groupby('low').inc.transform(lambda v:med(v,x.wt.loc[v.index]));x['mid']=m.adl.reindex(x.index)
  h=e.groupby('hh');ok=coverage(e)&e.inc.gt(0)&e['size'].eq(h.expense.transform('count'))
  x['oop']=h.expense.transform('sum').where(ok).reindex(x.index);x['den']=e.inc.reindex(x.index);x['gap']=(e.date-m.date).reindex(x.index)
  x['win']=w;x['wt']=e.wt.reindex(x.index);x['id']=x.index
  if name=='HRS':
   x['margin']=(h.start.transform('min').where(h.start.transform('count').eq(e['size']))-m.date).reindex(x.index)
  f=x[x.mid.notna()];f=f[e.date.reindex(f.index).notna()] if name=='HRS' else f[f.index.isin(e.index)]
  g=f[f.oop.notna()&f.den.gt(0)&f.wt.gt(0)].copy();g['mid']=g.mid.astype(int)
  fs.append([name,w,len(x),x.mid.notna().sum(),len(f),len(g),g.mid.sum()])
  rs.append(g)
 return pd.concat(rs),fs
def model(g,th,cluster):
 x=np.column_stack([np.ones(len(g)),g.low,g.mid,g.low*g.mid,g.age,g.female,pd.get_dummies(g.win,drop_first=True)]).astype(float)
 w=g.wt.to_numpy();y=(g.oop>th*g.den).to_numpy(float);inv=np.linalg.inv(x.T@(w[:,None]*x));b=inv@x.T@(w*y)
 sc=pd.DataFrame(x*(w*(y-x@b))[:,None]).groupby(g[cluster].to_numpy()).sum().to_numpy()
 M=len(sc);n,k=x.shape;cov=inv@sc.T@sc@inv*M/(M-1)*(n-1)/(n-k)
 hi=np.average(x,axis=0,weights=w);hi[1:4]=[0,1,0];lo=hi.copy();lo[1:4]=[1,1,1]
 for key,v in [('lower risk',lo),('higher risk',hi),('difference',lo-hi),('interaction',np.eye(k)[3])]:
  est=100*v@b;se=100*np.sqrt(v@cov@v);yield [th,key,est,est-1.96*se,est+1.96*se]
if __name__=='__main__':
 import sys
 for name in sys.argv[1:] or ['KLoSA','HRS']:
  waves,windows=(usa if name=='HRS' else korea)();g,fs=build(name,waves,windows);g.to_pickle(out/(name+'_temp.pkl'))
  pd.DataFrame(fs,columns=['cohort','baseline_wave','baseline','middle_ADL','followup','model','onset']).to_csv(out/(name+'_flow.csv'),index=False)
  allrows=[]
  specs={'reconstructed':g}
  if name=='HRS':specs['no household period starts before middle interview month']=g[g.margin.ge(0)]
  for label,z in specs.items():
   for th in [.05,.1,.25]:
    for row in model(z,th,'hh' if name=='HRS' else 'id'): allrows.append([label,len(z),int(z.mid.sum())]+row)
  a=pd.DataFrame(allrows,columns=['specification','person_windows','new_limitations','threshold','metric','value','lower95','upper95'])
  a.to_csv(out/(name+'_estimates.csv'),index=False);print(name,a.query('threshold==.1').round(2).to_string(index=False))
