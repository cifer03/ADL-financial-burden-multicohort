"""Run on the extracted UKDS 5050 Stata folder. Print aggregate results only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
adl = ['headldr', 'headlba', 'headlea', 'headlbe', 'headlwc']

def read(name, columns):
    paths = list(root.rglob(name + '.dta'))
    if len(paths) != 1:
        raise ValueError(f'Expected one {name}.dta, found {len(paths)}')
    frame = pd.read_stata(paths[0], columns=['idauniq', *columns], convert_categoricals=False)
    return frame.set_index('idauniq', verify_integrity=True)

base = read('wave_9_elsa_data_eul_v2', ['indager', 'indsex', 'idahhw9', *adl])
base['age'] = base.indager.replace(-7, 90)
base = base.loc[(base.age >= 60) & base[adl].eq(0).all(axis=1)].copy()
print('Age and ADL eligible:', len(base))
base = base.join(read('wave_9_financial_derived_variables', ['eqtotinc_bu_s', 'eqtotinc_bu_f']))
base = base.loc[base.eqtotinc_bu_f.eq(0) & base.eqtotinc_bu_s.notna()].copy()
threshold = base.eqtotinc_bu_s.median()
base['low'] = (base.eqtotinc_bu_s <= threshold).astype(int)
print('Valid baseline income:', len(base), 'median:', threshold)
mid = read('wave_10_elsa_data_eul_v4', adl)
mid = mid.loc[mid.isin([0, 1]).all(axis=1)]
base = base.join(mid.max(axis=1).rename('onset'))
outcomes = ['capsacc4', 'capsacc5']
base = base.join(read('wave_11_elsa_data_eul', outcomes))
for group, rows in base.groupby('low'):
    print('low=', group, 'baseline=', len(rows), 'observed wave 10=', rows.onset.notna().sum())
rows=[]
for outcome in outcomes:
    data = base.loc[base.onset.notna() & base[outcome].isin([0, 1])].copy()
    print(outcome, 'complete:', len(data), 'incident:', int(data.onset.sum()))
    print(data.groupby(['onset','low'])[outcome].agg(['count','mean']))
    # Income difference among incident cases, adjusted for baseline age and sex.
    cases = data.loc[data.onset.eq(1)].copy()
    assert cases.indsex.isin([1,2]).all() and cases.idahhw9.gt(0).all()
    X = np.column_stack([np.ones(len(cases)), cases.low, cases.age - 70, cases.indsex.eq(2)])
    y = cases[outcome].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    meat = pd.DataFrame(X * (y - X @ b)[:,None]).groupby(cases.idahhw9.to_numpy()).sum().to_numpy()
    bread = np.linalg.inv(X.T @ X)
    g, n, k = len(meat), *X.shape
    se = np.sqrt((bread @ meat.T @ meat @ bread)[1,1] * g/(g-1) * (n-1)/(n-k))
    rows.append([outcome,'unweighted',len(cases),*(100*np.array([b[1],b[1]-1.96*se,b[1]+1.96*se]))])

# Two-stage follow-up weighting, conditional on the observed baseline income split.
base = base.join(read('wave_9_elsa_data_eul_v2', ['hehelf']))

observed_mid = base.onset.notna().to_numpy()
observed_end = base.capsacc4.isin([0, 1]).to_numpy()
assert np.array_equal(observed_end, base.capsacc5.isin([0, 1]))
incident_end = observed_mid & observed_end & base.onset.eq(1).to_numpy()
age = (base.age.to_numpy() - 70) / 10
health = base.hehelf.where(base.hehelf.between(1, 5))
print('Missing baseline health:', health.isna().sum())
design = np.column_stack([np.ones(len(base)), base.low, age, age**2,
                          base.indsex.eq(2), health.fillna(health.median()), health.isna()])
onset = base.onset[observed_mid].to_numpy()
later = np.column_stack([design[observed_mid], onset, onset * base.low[observed_mid]])
case_design = design[incident_end][:, [0, 1, 2, 4]]
events = base.loc[incident_end, outcomes].to_numpy()
households, cluster = np.unique(base.idahhw9, return_inverse=True)
assert households.min() > 0

def probability(x, y, freq):
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(x @ beta)))
        step = np.linalg.solve(x.T @ ((freq * p * (1-p))[:, None] * x),
                               x.T @ (freq * (y-p)))
        beta += step
        if abs(step).max() < 1e-8:
            return 1 / (1 + np.exp(-(x @ beta)))
    raise RuntimeError('Follow-up model did not converge')

def estimate(freq):
    p1 = probability(design, observed_mid, freq)
    p2 = probability(later, observed_end[observed_mid], freq[observed_mid])
    weights = 1 / (p1[incident_end] * p2[incident_end[observed_mid]])
    w = weights * freq[incident_end]
    b = np.linalg.solve(case_design.T @ (w[:, None] * case_design),
                        case_design.T @ (w[:, None] * events))
    return 100 * b[1], weights

point, weights = estimate(np.ones(len(base)))
rng = np.random.default_rng(20261122)
replicates = []
for _ in range(1000):
    counts = np.bincount(rng.integers(len(households), size=len(households)), minlength=len(households))
    replicates.append(estimate(counts[cluster])[0])
print('Weighted differences (pp):', point, 'percentile limits:', np.percentile(replicates,[2.5,97.5],axis=0))
print('Weight range:', weights.min(), weights.max(), 'effective N:', weights.sum()**2 / (weights @ weights))

interval=np.percentile(replicates,[2.5,97.5],axis=0)
for j,outcome in enumerate(outcomes):
    rows.append([outcome,'followup_weighted',int(incident_end.sum()),point[j],interval[0,j],interval[1,j]])
pd.DataFrame(rows,columns=['outcome','model','n','estimate','lower95','upper95']).to_csv(sys.argv[2],index=False)
