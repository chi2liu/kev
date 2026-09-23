"""Markdown tables for the round-6 write-up from runs/r6-readout/<size>.json (scripts/round6_readout.py output).

    uv run python scripts/round6_tables.py            # prints one table per size
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import read_json  # noqa: E402

f = lambda b, pp=True: ("unread" if b == "unread" else (f"{100 * b['delta']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]" if pp else f"{b['delta']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"))


def table(size):
    d = read_json(f"runs/r6-readout/{size}.json")
    print(f"\n**{size}** (parent {d['parent']}, T {d['parent_temperature']:.2f}; pp unless stated; every interval is candidate minus parent)\n")
    print("| arm | long (selection panel) | short acc | short Brier | confident errors | SemIf | scienthoon | WANLI-v2 | TypeSafe | pooled external | verdict (failed criteria) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in d["arms"]:
        ext = a["externals"]; failed = [k.split("_", 1)[1] for k, v in a["criteria"].items() if not v]
        verdict = "**passes selection**" if a["passed"] else ("incomplete" if not a["complete"] else "fails: " + ", ".join(failed))
        print(f"| {a['arm']} | {f(a['long'])} | {f(a['short']['acc'])} | {f(a['short']['brier'], False)} | {f(a['short']['confident_error_rate'])} | "
              + " | ".join(f(ext[s]) for s in ("semif", "scienthoon", "wanli2", "typesafe")) + f" | {f(a['pooled_external'])} | {verdict} |")
    print("\nranking:", d["ranking"] or "none")


if __name__ == "__main__":
    for size in sys.argv[1:] or ("9b", "4b", "08b"):
        table(size)
