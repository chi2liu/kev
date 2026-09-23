"""Round 7 (PLAN.md, documents delta): rules 1-4 on saved rows for every arm against its own parent; Jev reported.
Committed before any round-7 read.

    uv run python scripts/round7_readout.py --out runs/r7-readout

Arms: runs/r7-docs/{00..03}-trial-* (9B seed 1, 9B seed 2, 4B, 0.8B) and runs/r7-docs-27b/00-trial-0, each read as
runs/r7-<arm>-{docs,semif,scienthoon,wanli2,typesafe,v9}; short = the trial's own transfer/rows.json. Parents: the released
checkpoints (runs/docs1-P*, runs/r5r-P*-<suite>, runs/r6-P*-wanli2, their trials' transfer rows) and, for the 27B arm, round-6
trial A (runs/docs1-27b-A, runs/r6-27b-A-<suite>). Every arm is served at the temperature fitted on its own development rows.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served, unknowable_report  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import EXTERNALS, MARGIN, boot, knowable, serve  # noqa: E402

PARENTS = {   # tag: (trial, {suite: read dir})
    "P9": ("runs/night2-9b-du/00-trial-0", {"docs": "runs/docs1-P9", "semif": "runs/r5r-P9-semif", "scienthoon": "runs/r5r-P9-scienthoon", "wanli2": "runs/r6-P9-wanli2", "typesafe": "runs/r5r-P9-typesafe"}),
    "P4": ("runs/night2-4b-du/00-trial-0", {"docs": "runs/docs1-P4", "semif": "runs/r5r-P4-semif", "scienthoon": "runs/r5r-P4-scienthoon", "wanli2": "runs/r6-P4-wanli2", "typesafe": "runs/r5r-P4-typesafe"}),
    "P08": ("runs/night2-08b-du2/00-trial-0", {"docs": "runs/docs1-P08", "semif": "runs/r5r-P08-semif", "scienthoon": "runs/r5r-P08-scienthoon", "wanli2": "runs/r6-P08-wanli2", "typesafe": "runs/r5r-P08-typesafe"}),
    "27b-A": ("runs/r6-27b/00-trial-0", {"docs": "runs/docs1-27b-A", **{s: f"runs/r6-27b-A-{s}" for s in EXTERNALS}}),
}
ARMS = {"9b-s1": ("runs/r7-docs/00-trial-0", "P9"), "9b-s2": ("runs/r7-docs/01-trial-1", "P9"), "4b-s1": ("runs/r7-docs/02-trial-2", "P4"),
        "08b-s1": ("runs/r7-docs/03-trial-3", "P08"), "27b-s1": ("runs/r7-docs-27b/00-trial-0", "27b-A")}
JEV = "runs/jev-documents-v1/rows.json"


def rows_at(trial, reads, t):
    return {"short": knowable(serve(trial, Path(trial) / "transfer/rows.json", t)[0]), **{s: knowable(serve(trial, f"{d}/rows.json", t)[0]) for s, d in reads.items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    jev = [r for r in read_json(JEV) if r["variant"] == "clean"]
    report = {"jev_docs_acc": metrics(jev)["acc"], "arms": {}}
    for arm, (trial, ptag) in ARMS.items():
        if not (Path(trial) / "transfer/rows.json").exists() or not Path(f"runs/r7-{arm}-docs/rows.json").exists():
            report["arms"][arm] = "not read yet"; continue
        ptrial, preads = PARENTS[ptag]
        pt, t = (served(read_json(Path(x) / "development/rows.json"), [])[0] for x in (ptrial, trial))
        P = rows_at(ptrial, preads, pt)
        C = rows_at(trial, {s: f"runs/r7-{arm}-{s}" for s in ("docs", *EXTERNALS)}, t)
        rep = {"trial": trial, "parent": ptrial, "temperature": t, "parent_temperature": pt,
               "docs": boot(C["docs"], P["docs"], "acc"), "docs_acc": metrics(C["docs"])["acc"], "parent_docs_acc": metrics(P["docs"])["acc"],
               "docs_vs_jev": boot(C["docs"], jev, "acc"), "docs_brier": metrics(C["docs"])["brier"],
               "short": {m: boot(C["short"], P["short"], m) for m in ("acc", "brier", "confident_error_rate")},
               "externals": {s: {**boot(C[s], P[s], "acc"), "n": len(C[s]), "margin": MARGIN(len(C[s]))} for s in EXTERNALS},
               "pooled_external": boot([r for s in EXTERNALS for r in C[s]], [r for s in EXTERNALS for r in P[s]], "acc"),
               "unknowable_share": unknowable_report(serve(trial, f"runs/r7-{arm}-v9/rows.json", t)[0])["share_at_0_9"]}
        s = rep["short"]
        rep["criteria"] = {"1_docs_lower_above_0": rep["docs"]["ci95"][0] > 0,
                           "2_short_acc_lower_at_least_minus_1pp": s["acc"]["ci95"][0] >= -0.01, "2_short_brier_upper_at_most_0.01": s["brier"]["ci95"][1] <= 0.01,
                           "2_short_confident_errors_upper_at_most_1pp": s["confident_error_rate"]["ci95"][1] <= 0.01,
                           **{f"3_{k}_lower_at_least_margin": v["ci95"][0] >= v["margin"] for k, v in rep["externals"].items()},
                           "3_pooled_lower_at_least_minus_1.5pp": rep["pooled_external"]["ci95"][0] >= -0.015, "3_unknowable_at_most_0.05": rep["unknowable_share"] <= 0.05}
        rep["passed"] = all(rep["criteria"].values())
        report["arms"][arm] = rep
    for size in ("9b", "4b", "08b", "27b"):
        passing = [(k, v) for k, v in report["arms"].items() if k.startswith(size + "-") and isinstance(v, dict) and v["passed"]]
        report[f"candidate_{size}"] = max(passing, key=lambda kv: kv[1]["docs"]["delta"])[0] if passing else None
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "round7.json", report)
    f = lambda b: f"{100 * b['delta']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]"
    print(f"Jev documents-v1 dev acc {report['jev_docs_acc']:.3f}")
    for arm, r in report["arms"].items():
        if not isinstance(r, dict): print(f"{arm:7} {r}"); continue
        print(f"{arm:7} docs {r['parent_docs_acc']:.3f} -> {r['docs_acc']:.3f} {f(r['docs'])} vs Jev {f(r['docs_vs_jev'])} | short acc {f(r['short']['acc'])} | "
              f"ext " + " ".join(f"{k}:{100 * v['delta']:+.1f}[{100 * v['ci95'][0]:+.1f}]" for k, v in r["externals"].items()) + f" pooled {f(r['pooled_external'])} | unk {r['unknowable_share']:.3f} "
              f"-> {'PASS' if r['passed'] else 'fail: ' + ', '.join(k for k, v in r['criteria'].items() if not v)}")
    print({k: v for k, v in report.items() if k.startswith("candidate_")})


if __name__ == "__main__":
    main()
