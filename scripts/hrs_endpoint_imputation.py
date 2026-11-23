"""Audit HRS endpoint spending flags from the authorised RAND 2022 v2 input."""
import pandas as pd
from rebuild import usa, build, coverage, med, model, out


def main():
    waves, windows = usa()
    panel, _ = build("HRS", waves, windows)
    fields = ["hhidpn"] + [f"r{w + 2}oopmdof" for w in windows]
    raw = pd.concat(pd.read_stata(out / "randhrs1992_2022v2.dta", columns=fields,
                                  chunksize=1000, convert_categoricals=False)).set_index("hhidpn")
    stages, parts = [], []
    for w in windows:
        b, m, e = (waves[i] for i in (w, w + 1, w + 2))
        flag = raw[f"r{w + 2}oopmdof"]
        no_imputation = flag.eq(0).groupby(e.hh).transform("all")
        base = b.loc[b.age.between(60, 120) & b.adl.eq(False) & b.inc.gt(0) &
                     b["size"].gt(0) & b.wt.gt(0) & coverage(b)]
        income = base.inc / base["size"].pow(0.5)
        low = income.le(med(income, base.wt))
        usable = (coverage(e) & e.inc.gt(0) &
                  e["size"].eq(e.expense.groupby(e.hh).transform("count")))
        for keep_name, keep in [("RAND available total", usable),
                                ("all members have unflagged totals", usable & no_imputation)]:
            eligible = (m.adl.notna() & e.date.notna() & keep).reindex(base.index)
            for group, mask in [("lower", low), ("higher", ~low)]:
                f = eligible & mask
                v = f & e.wt.reindex(base.index).gt(0)
                stages.append([w, group, keep_name, int(f.sum()), int(v.sum()),
                               int((v & m.adl.reindex(base.index).eq(True)).sum())])
        p = panel.loc[panel.win.eq(w)].copy()
        all_known = flag.isin([0, 1]).groupby(e.hh).transform("all")
        assert all_known.reindex(p.index).all()
        parts.append(p.assign(unflagged=no_imputation.reindex(p.index)))
    stages = pd.DataFrame(stages, columns=["baseline_wave","income","rule","finances","model","new_limits"])
    stages.to_csv(out / "hrs_endpoint_flag_flow.csv", index=False)
    p = pd.concat(parts)
    results = []
    for period, sub in [("seven", p), ("two", p.loc[p.win.ge(13)])]:
        for rule, g in [("RAND available total", sub), ("all members have unflagged totals", sub.loc[sub.unflagged])]:
            counts = stages.loc[stages.rule.eq(rule) & stages.baseline_wave.isin(g.win.unique())]
            assert len(g) == counts.model.sum() and g.mid.sum() == counts.new_limits.sum()
            contrast = next(r for r in model(g, .10, "hh") if r[1] == "difference")
            results.append([period, rule, len(g), g.id.nunique(), int(g.mid.sum()), *contrast[2:]])
    result = pd.DataFrame(results, columns=["windows","rule","observations","persons","new_limits","gap_pp","lower95","upper95"])
    result.to_csv(out / "hrs_endpoint_flag_models.csv", index=False)
    print(result.to_string(index=False))
    print(stages.loc[stages.baseline_wave.ge(13)].groupby(["rule","income"])[["finances","model","new_limits"]].sum())


if __name__ == "__main__":
    main()
