"""Assess the impact of treating public-health-centre-only visits as an unpriced category."""
import io,zipfile,pandas as pd
from rebuild import korea,build,model
waves,windows=korea()
z=zipfile.ZipFile('klosa_work/STATA.zip')
for w,e in waves.items():
 p=f'w{w:02}';r=pd.io.stata.StataReader(io.BytesIO(z.read(next(n for n in z.namelist() if n.startswith('str'+p[1:])))),convert_categoricals=False)
 fields=r.variable_labels()
 wt=next(p+k for k in ('wgt_ap','wgt_p','wgt_c') if p+k in fields)
 e['wt']=r.read(columns=['pid',wt]).set_index('pid')[wt]
parts=[]
for w in windows:
 e=waves[w+2].copy();q=waves.copy()
 # Only visits to the public health centre, with all four priced visit counts explicitly zero.
 centre_only=e.C333.gt(0)&e[['C330','C334','C337','C340']].eq(0).all(axis=1)
 inpatient=e.oopm1.where(e.oopm1.ge(0)).mask(e.oopm1.isna()&e.C318.eq(0),0)
 e['expense']=e.expense.mask(centre_only&e.oopm2.isna(),inpatient/2)
 q[w+2]=e
 a,_=build('KLoSA',q,[w]);parts.append(a)
full=pd.concat(parts)
print('model, onset, unique',len(full),full.mid.sum(),full.id.nunique());print(full.groupby('low').size());print(list(model(full,.1,'id')))
pd.DataFrame(list(model(full,.1,'id')),columns=['threshold','metric','estimate','lower95','upper95']).assign(windows=len(full),onset=full.mid.sum(),people=full.id.nunique()).to_csv('review/centre_only_scenario.csv',index=False)
