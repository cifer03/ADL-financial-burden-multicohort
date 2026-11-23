"""Korea Health Panel income-by-onset interactions across consecutive interview windows."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import sys
import pandas as pd
import numpy as np
from model import interaction
root=Path(sys.argv[1])

def read(archive, member):
    q=pd.read_sas(BytesIO(archive.read(member+'.sas7bdat')),format='sas7bdat')
    q.columns=q.columns.str.upper()
    return q

def wave(year):
    with ZipFile(root/f'KHP+{year}_sas.zip') as z:
        if year<2019:
            tag=f't{year%100:02}'
            d=read(z,tag+'ind')[['PIDWON','HHID','C3','C4_0']].rename(columns={'C3':'sex','C4_0':'birth'})
            h=read(z,tag+'hh')[['HHID','TOTAL','B1']].rename(columns={'TOTAL':'income','B1':'size'})
            a=read(z,tag+'appen').set_index('PIDWON',verify_integrity=True)
            d=d.set_index('PIDWON',verify_integrity=True).join(a.reindex(columns=['SJ3','SE1','SE2']))
            d['activity']=d.SJ3
            d['outcome']=d.SE2.eq(1).where(d.SE1.eq(1)&d.SE2.gt(0)).mask(d.SE1.isin([2,3]),False)
            d['weight']=1.
        else:
            tag=chr(ord('a')+year-2019)
            a=read(z,tag+'_ind').set_index('PIDWON',verify_integrity=True)
            d=a[['HHID','H_INC_TOT','SEX','BIRTH_Y','HS_EQ3']].rename(columns={'H_INC_TOT':'income','SEX':'sex','BIRTH_Y':'birth','HS_EQ3':'activity'}).copy()
            h=read(z,tag+'_hh')[['HHID','FAM_N']].rename(columns={'FAM_N':'size'})
            access=a.get('AH1',pd.Series(np.nan,index=a.index))
            reason=a.get('AH1_1_D',pd.Series(np.nan,index=a.index))
            d['outcome']=reason.eq(1).where(access.eq(1)&reason.isin([1,2])).mask(access.isin([2,3]),False)
            d['weight']=a.get('I_WGL',np.nan)
        d=d.join(h.set_index('HHID',verify_integrity=True),on='HHID')
        d['age']=year-d.birth
        return d

def run(starts,weighted):
    waves={yr:wave(yr) for yr in sorted({w+j for w in starts for j in range(3)})}
    parts=[]
    for year in starts:
        b,m,e=(waves[year+i] for i in range(3))
        x=b.loc[b.age.between(60,120)&b.activity.eq(1)&b.income.gt(0)&b['size'].gt(0)].copy()
        eq=x.income/x['size'].pow(.5)
        x['low']=eq.le(eq.median())
        x['onset']=m.activity.gt(1).where(m.activity.isin([1,2,3]))
        x['outcome']=e.outcome
        x['weight']=e.weight
        x=x.dropna(subset=['onset','outcome','sex'])
        if weighted:
            x=x[x.weight.gt(0)].copy()
            x['weight']=x.weight/x.weight.sum()
        x['batch']=year
        parts.append(x)
    g=pd.concat(parts);g['onset']=g.onset.astype(int);g['participant']=g.index;g=g.reset_index(drop=True)
    return [len(g),g.participant.nunique(),*interaction(g,g.outcome,g.participant,g.weight if weighted else None)]
rows=[['first_panel',*run([2009,2010,2011,2012,2015,2016],False)],
      ['second_panel',*run([2019,2020,2021,2022],True)]]
pd.DataFrame(rows,columns=['outcome','n','persons','estimate','lower95','upper95']).to_csv(sys.argv[2],index=False)
