"""Paired 2018 to wave-6 spending burden. Supply the directory of original ZIPs."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import sys
import pandas as pd
import numpy as np
from model import interaction
root=Path(sys.argv[1])

def read(archive, module, columns, key='ID'):
    with ZipFile(root/archive) as z:
        name=next(n for n in z.namelist() if Path(n).name==module+'.dta')
        return pd.read_stata(BytesIO(z.read(name)),columns=[key,*columns],convert_categoricals=False).set_index(key,verify_integrity=True)

def budget(archive,baseline):
    if baseline:
        weekly=['ge006_w4','ge006_w2_1','ge007_w4','ge008']
        monthly=[f'ge009_{i}' for i in range(1,8)]
        annual=[f'ge010_{i}' for i in [*range(1,10),*range(11,16)]]
        skip='ge006_w2'; medical='ge010_6';module='Household_Income'
    else:
        weekly=['ge005','ge007','ge008','ge009']
        monthly=[f'ge010_{i}' for i in range(1,8)]
        annual=[f'ge012_{i}' for i in [*range(1,5),*range(6,11),*range(15,20)]]
        skip='ge006';medical='ge012_7';module='Household_INC_EXP_Assets'
    q=read(archive,module,[*weekly,*monthly,*annual,skip],key='householdID')
    q[weekly[1]]=q[weekly[1]].mask(q[weekly[1]].isna() & q[skip].eq(2),0)
    def total(cols):
        return q[cols].where(q[cols].ge(0)).sum(axis=1,min_count=len(cols))
    den=52*total(weekly)+12*total(monthly)+total(annual)
    return q[medical].div(den.where(den.gt(0)))

def adl(archive,items):
    x=read(archive,'Health_Status_and_Functioning',items)
    return x.gt(1).any(axis=1).where(x.isin([1,2,3,4]).all(axis=1))

def month(s):
    return 12*pd.to_numeric(s.iyear)+pd.to_numeric(s.imonth)

b='CHARLS2018r.zip';m='CHARLS2020r.zip'
x=read(b,'Sample_Infor',['householdID','communityID','iyear'])
dem=read(b,'Demographic_Background',['ba002_1','ba004_w3_1','xrgender'])
x['age']=pd.to_numeric(x.iyear)-dem.ba002_1.combine_first(dem.ba004_w3_1)
x['sex']=dem.xrgender
x=x.loc[x.age.between(60,120)&adl(b,[f'db{i:03}' for i in range(10,15)]).eq(False)].copy()
inc=read('H_CHARLS_D_Data.zip','H_CHARLS_D_Data',['hh4itot','h4hhres'])
inc.index=inc.index.astype(str);x=x.join(inc)
x=x[x.hh4itot.gt(0)&x.h4hhres.gt(0)].copy()
eq=x.hh4itot/x.h4hhres.pow(.5);x['low']=eq.le(eq.median())
x['before']=x.householdID.map(budget(b,True))
x['onset']=adl(m,[f'db{i:03}' for i in [1,3,5,7,9]])
earlier=month(read(m,'Sample_Infor',['iyear','imonth']))
ends=[]
for year in [2021,2022,2023]:
    a=f'CHARLS-W6-{year}.zip'
    s=read('CHARLS-W6-2023.zip','Sample_Infor',['householdID','res_wave','died','iyear','imonth'])
    s=s.loc[pd.to_numeric(s.res_wave).eq(year)&s.died.eq(0)].copy()
    s['after']=s.householdID.map(budget(a,False))
    s['gap']=month(s)-earlier
    ends.append(s.assign(batch=year)[['after','gap','batch']])
x=x.join(pd.concat(ends,verify_integrity=True)).dropna(subset=['before','after','onset','age','sex','communityID'])
x=x[x.gap.ge(13)].copy()
x['onset']=x.onset.astype(int)
print('N, onset, communities',len(x),x.onset.sum(),x.communityID.nunique())
rows=[]
for threshold in [.1,.25]:
    delta=x.after.gt(threshold).astype(int)-x.before.gt(threshold).astype(int)
    rows.append([f'budget_{threshold:.0%}',len(x),int(x.onset.sum()),*interaction(x,delta,x.communityID,use_t=True)])
pd.DataFrame(rows,columns=['outcome','n','new_limitations','estimate','lower95','upper95']).to_csv(sys.argv[2],index=False)
