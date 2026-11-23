"""Weighted linear probability contrast with finite-sample cluster correction."""
import numpy as np
import pandas as pd
from scipy.stats import t, norm

def interaction(frame, outcome, cluster, weights=None, use_t=False):
    x = pd.DataFrame({'constant':1., 'low':frame.low,'onset':frame.onset,
                      'interaction':frame.low*frame.onset, 'age':frame.age,'sex':frame.sex})
    x = x.join(pd.get_dummies(frame.batch, drop_first=True)).to_numpy(float)
    w = np.ones(len(frame)) if weights is None else np.asarray(weights)
    y = np.asarray(outcome, dtype=float)
    bread = np.linalg.inv(x.T @ (w[:,None]*x))
    beta = bread @ x.T @ (w*y)
    scores = pd.DataFrame(x*(w*(y-x@beta))[:,None]).groupby(np.asarray(cluster)).sum()
    n,k=x.shape
    m=len(scores)
    se=np.sqrt((bread @ scores.to_numpy().T @ scores.to_numpy() @ bread)[3,3]*m/(m-1)*(n-1)/(n-k))
    critical=t.ppf(.975,m-1) if use_t else norm.ppf(.975)
    return 100*np.array([beta[3],beta[3]-critical*se,beta[3]+critical*se])
