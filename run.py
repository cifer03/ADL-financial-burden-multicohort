"""Run selected v35 analyses in a private working directory."""
import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'
REFERENCE = ROOT / 'reference'
JOBS = {
    'hrs': [('rebuild.py', 'HRS'), ('hrs_candidate.py',), ('hrs_endpoint_imputation.py',), ('hrs_slots.py',)],
    'share': [('share.py',), ('share_slots.py',)],
    'klosa': [('klosa_income_gap.py',), ('klosa_slots.py',)],
}
OUTPUTS = {
    'hrs': ['review/HRS_flow.csv', 'review/hrs_candidate_models.csv', 'review/hrs_endpoint_flag_models.csv',
            'review/hrs_slots.csv', 'review/hrs_slots_temporal.csv'],
    'share': ['rerun/SHARE_estimates.csv', 'review/share_slots.csv'],
    'klosa': ['review/centre_only_scenario.csv', 'review/klosa_slots.csv'],
}


def compare(path):
    actual = pd.read_csv(path)
    reference = pd.read_csv(REFERENCE / path.name)
    pd.testing.assert_frame_equal(actual, reference, check_exact=False, atol=1e-8, rtol=1e-8)


def figures(work, source):
    from scripts.income_gap_figure import draw
    share_source = REFERENCE if source == REFERENCE else work / 'rerun'
    draw(source / 'hrs_candidate_models.csv', share_source / 'SHARE_estimates.csv',
         source / 'centre_only_scenario.csv', work / 'review/figure_1.pdf')
    rows = []
    for cohort in ('HRS', 'SHARE', 'KLoSA'):
        table = pd.read_csv(source / (cohort.lower() + '_slots.csv'))
        rows.append(table.assign(cohort=cohort))
    results = pd.concat(rows, ignore_index=True)
    results.to_csv(work / 'review/three_cohort_slots.csv', index=False)
    total = results.loc[results.assessed_per_1000_by_group.eq('all cases')]
    table = total.pivot(index='rule', columns='cohort', values='estimate')
    table.loc[['spending', 'spending divided by income-group median', 'lower-income group first',
               'contemporaneous actual ratio'], ['HRS', 'SHARE', 'KLoSA']].to_csv(work / 'review/table_2.csv', float_format='%.1f')
    subprocess.run([sys.executable, str(SCRIPTS / 'policy_figure.py')], cwd=work, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=[*JOBS, 'all', 'from-reference'])
    parser.add_argument('--work', type=Path, required=True, help='private work directory containing licensed inputs')
    args = parser.parse_args()
    work = args.work.resolve()
    for folder in ['review', 'rerun']:
        (work / folder).mkdir(parents=True, exist_ok=True)
    if args.target == 'from-reference':
        figures(work, REFERENCE)
        return
    for cohort in JOBS if args.target == 'all' else [args.target]:
        for script, *params in JOBS[cohort]:
            subprocess.run([sys.executable, str(SCRIPTS / script), *params], cwd=work, check=True)
        for name in OUTPUTS[cohort]:
            compare(work / name)
        print(f'{cohort}: reference checks passed', flush=True)
    if args.target == 'all':
        figures(work, work / 'review')


if __name__ == '__main__':
    main()
