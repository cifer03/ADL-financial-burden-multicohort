# Income differences in medical spending burden after newly reported limitations in daily living

Analysis code for manuscript v35. The source release covers the HRS and SHARE income-gap estimates, the three-cohort retrospective allocation analysis, selected HRS sensitivity and temporal analyses, and all three complementary surveys in figure 3.

## Install and run

Use Python 3.12 (tested with 3.12.14). Run these commands from the extracted code directory.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run.py from-reference --work ../figure-work
```

The last command uses the included cohort aggregates to generate review/figure_1.pdf, review/figure_2.pdf and review/table_2.csv under the indicated work directory. On Windows activate the environment using .venv\Scripts\activate.

## Reproduce from cohort inputs

Obtain authorised distributions from the corresponding study custodian. Create an empty private work directory outside the code repository. Place these five original ZIP files under it, preserving the names and subdirectories below. For a single-cohort run only that cohort's inputs are needed.

| Cohort | Paths relative to the private work directory |
|---|---|
| RAND HRS 1992–2022 V2 | check/randhrs1992_2022v2_STATA.zip |
| SHARE release 9.0.0 | sharework/sharew5_rel9-0-0_ALL_datasets_stata.zip, sharework/sharew6_rel9-0-0_ALL_datasets_stata.zip, sharework/sharew7_rel9-0-0_ALL_datasets_stata.zip |
| KLoSA | klosa_work/STATA.zip, containing the ten wave-specific str Stata files |

```sh
python run.py all --work ../private-work
```

Targets hrs, share and klosa run individual cohorts. Each target compares its listed aggregates to the reference CSVs, checking labels, row counts and numeric values. The all target then builds both figures and table 2. Outputs are written to review/ or rerun/ inside the work directory. Use a fresh work directory when changing the input release; HRS reuses the extracted Stata file on subsequent runs. HRS requires several GB of working space. The extracted dataset and intermediate participant-level pickle in the private work directory are subject to the source study’s access conditions.

## Coverage of manuscript results

| Result | Implementation and scope |
|---|---|
| Figure 1, HRS | rebuild.py constructs seven windows. hrs_candidate.py estimates the primary gap and the baseline-adjusted two-window sensitivity. |
| Figure 1, SHARE | share.py computes within-imputation estimates and pools the five imputations for waves 5–7. |
| Figure 1, KLoSA | klosa_income_gap.py fits the income-gap model on the same four-priced-category definition used for allocation. The rerun includes 21,274 person-windows. The historical v35 model included 21,255. Both estimates round to 7.2 percentage points (95% CI −0.2 to 14.6). Use the rerun output when synchronising manuscript counts. |
| Figure 2 and table 2 | hrs_slots.py, share_slots.py and klosa_slots.py implement all four ranking rules. Each allocates 15% within survey window or SHARE country. The population denominator is all eligible newly limited observations. KLoSA uses the separate four-priced-outpatient-category model (21,274 model observations, 528 newly limited). |
| HRS sensitivity and temporal validation | hrs_endpoint_imputation.py tests exclusion of households with flagged endpoint spending totals. hrs_slots.py invokes the disjoint-wave temporal analysis in timing_validation.py (table S7). |
| Additional material | Figure 3 is reproduced separately from CHARLS, ELSA and Korea Health Panel inputs using figure3.py. Table S6 and the full supplementary appendix are not covered end to end. |

The observed spending-to-income ratio benchmark uses the variables defining high burden. Retrospective allocation assumes follow-up expenditure is available. The HRS temporal analysis uses prior information and a separate sample.

The reference directory contains aggregate tables only. The code package contains no cohort microdata. The three-cohort reproduction audit is the separate internal record of earlier specifications.

## Figure 3 and complementary cohorts

The manuscript uses six surveys. HRS, SHARE and KLoSA inform the primary analyses. CHARLS, ELSA and the Korea Health Panel (KHP) inform Figure 3. KLoSA and KHP are separate studies. Use the following additional entry point to reproduce Figure 3 from authorised inputs.

```sh
python figure3.py --inputs ../complementary-zips --elsa ../ukds5050 --output ../figure3-output
```

Place H_CHARLS_D_Data.zip, CHARLS2018r.zip, CHARLS2020r.zip and CHARLS-W6-2021.zip through CHARLS-W6-2023.zip under the input directory. Place KHP+YYYY_sas.zip there for 2009–2018 and 2019–2024. Extract the UKDS 5050 wave 9, 10 and 11 core data and the wave 9 financial derived variables under the ELSA directory. The ELSA script checks that each required Stata file occurs exactly once.

* CHARLS computes annual comparable household budgets in 2018 and wave 6, links household expenditure separately at each interview, and estimates the income-by-onset contrast in individual change in burden status. The 2020 interview identifies onset; only follow-up interviews at least 13 months later contribute. Five common ADL items and positive baseline income and household size determine baseline eligibility. The paired model adjusts for baseline age, sex and fieldwork batch, clustering by baseline community.
* ELSA estimates baseline household-clustered income differences among participants with new difficulties and a valid endpoint, plus two-stage follow-up-weighted sensitivities using 1000 household resamples. Unmet GP or dental access includes any reason.
* KHP analyses first-panel windows starting in 2009, 2010, 2011, 2012, 2015 and 2016, and second-panel windows starting in 2019–2022. Positive endpoint longitudinal weights in the second panel are normalised to equal total weight within each window. Models cluster on participant. The first panel uses the principal reason for unmet care; the second uses the cost-specific reason indicator.

figure3.py writes one aggregate CSV per survey and a regenerated plot. It checks sample sizes and all reported estimates and interval endpoints at the precision of v35. Figure 3 uses three distinct estimands and has no pooled estimate. No participant-level inputs or derived records are included in the release.
