"""Render the three income-gap model outputs."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def draw(hrs_path, share_path, korea_path, destination):
    hrs = pd.read_csv(hrs_path).query("spec == 'seven windows' and metric == 'difference'").iloc[0]
    share = pd.read_csv(share_path).query("income_share_threshold == 0.1 and estimate == 'gap'").iloc[0]
    korea = pd.read_csv(korea_path).query("metric == 'difference' and threshold == 0.1").iloc[0]
    estimates = [('HRS', hrs.estimate, hrs.lower95, hrs.upper95),
                 ('SHARE', share.percent_or_pp, share.lower95, share.upper95),
                 ('KLoSA', korea.estimate, korea.lower95, korea.upper95)]
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.subplots_adjust(left=.12, right=.73, bottom=.22, top=.82)
    def label(v):
        return f'{v:.1f}'.replace('.', '·').replace('-', '−')
    for i, (name, estimate, lo, hi) in enumerate(estimates):
        ax.errorbar(estimate, 2-i, xerr=[[estimate-lo], [hi-estimate]], fmt='o', color='#205472', capsize=3)
        ax.text(1.03, 2-i, f'{label(estimate)} ({label(lo)} to {label(hi)})',
                transform=ax.get_yaxis_transform(), va='center', fontsize=10)
    ax.set_yticks([2, 1, 0], [r[0] for r in estimates])
    ax.set_ylim(-.6, 2.6)
    ax.axvline(0, color='gray', lw=.8)
    ax.set_xlabel('Lower-income minus higher-income risk (percentage points)')
    ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Income gap in high medical spending burden', fontweight='bold')
    fig.savefig(destination)
    plt.close(fig)
