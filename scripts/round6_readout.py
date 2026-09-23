"""Round-6 selection read-out (PLAN.md, Round 6, rules 1-4): every delta trial of a size against the released parent, from
saved rows, with the ranking that picks the confirmation candidate. Selection sets only.

    uv run python scripts/round6_readout.py --size 9b --out runs/r6-readout

Arms: the trials under runs/r6-<size>/ and runs/r6-<size>-r2/ (labelled from their config) plus round 5's incumbent (C9 / C4 / C08).
Rows: short = each trial's own transfer/rows.json (transfer-v4 development, written inside the trial) against the parent's;
long = runs/r6-<size>-<arm>-long (longstate-v2 development) against runs/r5r-P<size>-long; externals = runs/r6-<size>-<arm>-<suite>
for semif / scienthoon / wanli2 / typesafe and v9 against the parent's reads (runs/r5r-P<size>-<suite>, runs/r6-P<size>-wanli2).
A missing read leaves that criterion "unread" and the arm unranked. Every arm is served at the temperature fitted on its own
development rows (kev.metrics.served).
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, paired_bootstrap, raw_row, recorded, served, tempered_row, unknowable_report  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

SAMPLES = 2000
PARENTS = {"9b": ("P9", "runs/night2-9b-du/00-trial-0"), "4b": ("P4", "runs/night2-4b-du/00-trial-0"), "08b": ("P08", "runs/night2-08b-du2/00-trial-0")}
INCUMBENTS = {"9b": ("C9", "runs/r5-combined/00-trial-0", "r5r-C9"), "4b": ("C4", "runs/r5-combined/03-trial-3", "r5r-C4"), "08b": ("C08", "runs/r5-combined/05-trial-5", "r5r-C08")}
EXTERNALS = {"semif": 144, "scienthoon": 900, "wanli2": 1002, "typesafe": 89}
MARGIN = lambda n: -0.02 if n >= 500 else -0.03


def label(config):
    data = config.get("data", "")
    name = data.split("/")[-2] if data else "v7"
    if name == "combined-v1": name = "combined"
    extras = [f"replay{config['replay']}" for _ in [0] if config.get("replay", 2000) != 2000] + [f"lr{config['lr']:g}" for _ in [0] if config["lr"] not in (2e-05, 4e-05)] + [f"seed{config['seed']}" for _ in [0] if config.get("seed", 1) != 1]
    return "-".join([name, *extras])


def serve(trial, rows_path, t=None):
    """Clean rows of a read served at the trial's own development-fitted temperature (unknowable rows kept)."""
    if not Path(rows_path).exists(): return None, t
    if t is None: t = served(read_json(Path(trial) / "development/rows.json"), [])[0]
    return [tempered_row(raw_row(recorded(r)), t) for r in read_json(rows_path) if r["variant"] == "clean"], t


knowable = lambda rows, source=None: [r for r in rows if r["source"] != "unknowable" and (source is None or r["source"] == source)]


def boot(a, b, metric):
    x = paired_bootstrap(a, b, samples=SAMPLES, seed=0, metric=metric, aggregation="micro")
    return {"delta": x[f"micro_{metric}_delta"], "ci95": x["ci95"]}


def arm_report(name, trial, prefix, parent):
    """prefix: runs/<prefix>-<suite> read directories of this arm; parent: {suite: served knowable rows} of the parent."""
    rep, t = {"arm": name, "trial": trial}, None
    short, t = serve(trial, Path(trial) / "transfer/rows.json")
    rep["temperature"] = t
    rep["short"] = {m: boot(knowable(short), parent["short"], m) for m in ("acc", "brier", "confident_error_rate")}
    rep["short_metrics"] = {k: metrics(knowable(short))[k] for k in ("acc", "brier", "confident_error_rate", "coverage_at_5pct_error")}
    long, _ = serve(trial, f"runs/{prefix}-long/rows.json", t)
    rep["long"] = boot(knowable(long, "longstate"), parent["long"], "acc") if long else "unread"
    if long: rep["long_acc"] = metrics(knowable(long, "longstate"))["acc"]
    ext, pooled_c, pooled_p = {}, [], []
    for suite, n in EXTERNALS.items():
        rows, _ = serve(trial, f"runs/{prefix}-{suite}/rows.json", t)
        if rows is None or suite not in parent: ext[suite] = "unread"; continue
        c, p = knowable(rows), parent[suite]
        ext[suite] = {**boot(c, p, "acc"), "n": len(c), "margin": MARGIN(n), "acc": metrics(c)["acc"]}
        pooled_c += c; pooled_p += p
    rep["externals"] = ext
    rep["pooled_external"] = boot(pooled_c, pooled_p, "acc") if pooled_c and all(v != "unread" for v in ext.values()) else "unread"
    v9, _ = serve(trial, f"runs/{prefix}-v9/rows.json", t)
    rep["unknowable_share"] = unknowable_report(v9)["share_at_0_9"] if v9 else "unread"
    c = {}
    if rep["long"] != "unread":
        c["1_long_lower_above_0"] = rep["long"]["ci95"][0] > 0; c["1_long_point_at_least_5pp"] = rep["long"]["delta"] >= 0.05
    s = rep["short"]
    c.update({"2_short_acc_lower_at_least_minus_1pp": s["acc"]["ci95"][0] >= -0.01, "2_short_brier_upper_at_most_0.01": s["brier"]["ci95"][1] <= 0.01,
              "2_short_confident_errors_upper_at_most_1pp": s["confident_error_rate"]["ci95"][1] <= 0.01})
    for suite, e in ext.items():
        if e != "unread": c[f"3_{suite}_lower_at_least_margin"] = e["ci95"][0] >= e["margin"]
    if rep["pooled_external"] != "unread":
        c["3_pooled_point_at_least_0"] = rep["pooled_external"]["delta"] >= 0; c["3_pooled_lower_at_least_minus_1pp"] = rep["pooled_external"]["ci95"][0] >= -0.01
    if rep["unknowable_share"] != "unread": c["3_unknowable_share_at_most_0.05"] = rep["unknowable_share"] <= 0.05
    rep["criteria"] = c
    rep["complete"] = rep["long"] != "unread" and rep["pooled_external"] != "unread" and rep["unknowable_share"] != "unread"
    rep["passed"] = rep["complete"] and all(c.values())
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", required=True, choices=list(PARENTS)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ptag, ptrial = PARENTS[a.size]
    pt = served(read_json(Path(ptrial) / "development/rows.json"), [])[0]
    parent = {"short": knowable(serve(ptrial, Path(ptrial) / "transfer/rows.json", pt)[0]), "long": knowable(serve(ptrial, f"runs/r5r-{ptag}-long/rows.json", pt)[0], "longstate")}
    for suite in EXTERNALS:
        path = f"runs/r6-{ptag}-wanli2/rows.json" if suite == "wanli2" else f"runs/r5r-{ptag}-{suite}/rows.json"
        rows, _ = serve(ptrial, path, pt)
        if rows: parent[suite] = knowable(rows)
    arms = []
    itag, itrial, iprefix = INCUMBENTS[a.size]
    arms.append(arm_report(itag, itrial, iprefix, parent))
    for d in sorted(Path("runs").glob(f"r6-{a.size}/*-trial-*")) + sorted(Path("runs").glob(f"r6-{a.size}-r2/*-trial-*")):   # rounds 1 and 2
        if not (d / "transfer/rows.json").exists(): continue
        cfg = read_json(d / "provenance.json")["config"]
        arms.append(arm_report(label(cfg), str(d), f"r6-{a.size}-{label(cfg)}", parent))
    ranked = sorted([r for r in arms if r["passed"]], key=lambda r: (-r["long"]["delta"], r["short"]["brier"]["delta"], -r["pooled_external"]["delta"]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{a.size}.json", {"size": a.size, "parent": ptrial, "parent_temperature": pt, "arms": arms, "ranking": [r["arm"] for r in ranked]})
    f = lambda b: f"{b['delta']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]" if isinstance(b, dict) else b
    print(f"== {a.size} vs {ptag} (T {pt:.2f}); arms: {len(arms)}")
    for r in arms:
        ext = " ".join(f"{k}:{f(v)}" if v == 'unread' else f"{k}:{v['delta']:+.3f}[{v['ci95'][0]:+.3f}]" for k, v in r["externals"].items())
        print(f"  {r['arm']:26} long {f(r['long'])}  short acc {f(r['short']['acc'])} brier {f(r['short']['brier'])} conf {f(r['short']['confident_error_rate'])}\n"
              f"  {'':26} ext {ext}  pooled {f(r['pooled_external'])}  unk {r['unknowable_share']}  -> {'PASS' if r['passed'] else ('incomplete' if not r['complete'] else 'fail')}")
    print("ranking:", [r["arm"] for r in ranked] or "none passing yet")


if __name__ == "__main__":
    main()
