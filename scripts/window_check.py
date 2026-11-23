"""Window-level decomposition of the KLoSA allocation using review/rebuild.py."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd
from rebuild import korea, build

def allocated(score, weights):
    masses = weights.groupby(score).sum().sort_index(ascending=False)
    share = ((0.15 * weights.sum() - masses.cumsum().shift(fill_value=0)) / masses).clip(0, 1)
    selected = score.map(share)
    assert np.isclose(selected @ weights, 0.15 * weights.sum())
    return selected

def panel_windows():
    waves, windows = korea()
    with ZipFile('klosa_work/STATA.zip') as z:
        for w, frame in waves.items():
            prefix = f'w{w:02d}'
            name = next(n for n in z.namelist() if n.startswith(f'str{w:02d}'))
            with pd.io.stata.StataReader(BytesIO(z.read(name)), convert_categoricals=False) as reader:
                wt = next(prefix + k for k in ['wgt_ap','wgt_p','wgt_c'] if prefix + k in reader.variable_labels())
                frame['wt'] = reader.read(columns=['pid', wt]).set_index('pid')[wt]
    for w in windows:
        end = waves[w+2].copy()
        centre = end.C333.gt(0) & end[['C330','C334','C337','C340']].eq(0).all(axis=1)
        inpatient = end.oopm1.where(end.oopm1.ge(0)).mask(end.oopm1.isna() & end.C318.eq(0),0)
        end['expense'] = end.expense.mask(centre & end.oopm2.isna(), inpatient/2)
        tmp = waves.copy()
        tmp[w+2] = end
        panel, _ = build('KLoSA', tmp, [w])
        g = panel.loc[panel.mid.eq(1)]
        yield w, g

def run():
    rows = []
    for w,g in panel_windows():
        eligible = g.wt.sum()
        a = allocated(g.oop, g.wt)
        b = allocated(g.oop/g.ref, g.wt)
        for group, mask in [('higher',~g.low),('lower',g.low)]:
            cases = mask & g.oop.gt(0.10*g.den)
            rows.append([w,group,len(g),eligible,cases.sum(),g.wt[cases].sum(),(a*g.wt)[cases].sum(),(b*g.wt)[cases].sum()])
    result = pd.DataFrame(rows,columns=['baseline_wave','income_group','eligible_n','eligible_weight','case_n','case_weight','identified_spending_weight','identified_income_weight'])
    result.to_csv('review/klosa_by_window.csv',index=False)
    for group,g in result.groupby('income_group'):
        amounts=g[['case_weight','identified_spending_weight','identified_income_weight']]
        for style, pooled in [('equal windows',amounts.div(g.eligible_weight,axis=0).mean()),('pooled survey weights',amounts.sum()/g.eligible_weight.sum()),('window sizes',amounts.div(g.eligible_weight,axis=0).mul(g.eligible_n,axis=0).sum()/g.eligible_n.sum())]:
            den,a,b=pooled
            print(group,style,'per thousand:',np.round(1000*pooled,3),'miss difference:',round(100*(a-b)/den,3))
    print(result.query("income_group=='higher'")[['baseline_wave','eligible_n','case_n']].to_string(index=False))
if __name__=='__main__':
    run()
