"""Run from the audit project root. Outputs contain aggregated results only."""
import numpy as np
import pandas as pd
from rebuild import usa, build, model, out
from pooled_check import summarize
from timing_validation import get_prior


def fit(frame, adjust):
    design = pd.DataFrame({
        'constant': 1, 'lower_income': frame.low, 'limitation': frame.mid,
        'interaction': frame.low * frame.mid, 'age': frame.age, 'female': frame.female,
    }, index=frame.index)
    if adjust:
        design['baseline_burden'] = frame.prior.gt(.10)
    design = pd.concat([design, pd.get_dummies(frame.win, drop_first=True)], axis=1)
    x = design.to_numpy(float)
    w = frame.wt.to_numpy(float)
    y = frame.oop.gt(.10 * frame.den).to_numpy(float)
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    beta = bread @ (x.T @ (w * y))
    scores = pd.DataFrame(x * (w * (y - x @ beta))[:, None]).groupby(frame.hh.to_numpy()).sum()
    clusters = len(scores)
    n, k = x.shape
    covariance = bread @ (scores.to_numpy().T @ scores.to_numpy()) @ bread
    covariance *= clusters / (clusters - 1) * (n - 1) / (n - k)
    higher = np.average(x, axis=0, weights=w)
    higher[1:4] = [0, 1, 0]
    lower = higher.copy()
    lower[1:4] = [1, 1, 1]
    results = []
    for metric, contrast in [('lower risk', lower), ('higher risk', higher),
                             ('difference', lower-higher), ('interaction', np.eye(k)[3])]:
        estimate = 100 * contrast @ beta
        margin = 196 * np.sqrt(contrast @ covariance @ contrast)
        results.append([metric, estimate, estimate-margin, estimate+margin])
    if not adjust:
        reference = list(model(frame, .10, 'hh'))
        np.testing.assert_allclose([r[1:] for r in results], [r[2:] for r in reference], atol=1e-7)
    return pd.DataFrame(results, columns=['metric','estimate','lower95','upper95'])


def main():
    waves, windows = usa()
    g, _ = build('HRS', waves, windows)
    assert not g.duplicated(['win','id']).any()
    short = g.loc[g.win.ge(13)].copy()
    short['prior'] = np.nan
    for win, part in short.groupby('win'):
        short.loc[short.win.eq(win), 'prior'] = get_prior(waves[win]).reindex(part.index).to_numpy()
    complete = short.dropna(subset=['prior'])
    specs = [('seven windows',g,False), ('two windows',short,False),
             ('baseline complete',complete,False), ('baseline complete adjusted',complete,True)]
    models = pd.concat([fit(frame,adjust).assign(spec=name, person_windows=len(frame),
                        people=frame.id.nunique(), new_limitations=int(frame.mid.sum()))
                        for name,frame,adjust in specs],ignore_index=True)
    allocation = short.loc[short.mid.eq(1)]
    policy = summarize(allocation, 'hh').assign(person_windows=len(allocation),people=allocation.id.nunique())
    models.to_csv(out/'hrs_candidate_models.csv',index=False)
    policy.to_csv(out/'hrs_candidate_policy.csv',index=False)
    print(models.round(2).to_string(index=False), policy.round(2).to_string(index=False), sep='\n')


if __name__ == '__main__':
    main()
