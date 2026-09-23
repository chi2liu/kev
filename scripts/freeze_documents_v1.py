"""documents-v1: apply the B2 label rules to the candidates and the LLM answers (scripts/label_documents_v1.py), write the
adjudication queue and, once adjudications exist, the frozen suite.

    uv run python scripts/freeze_documents_v1.py --report                  # agreement tables, writes the adjudication queue
    uv run python scripts/freeze_documents_v1.py --combine                 # two independent adjudications -> adjudications.jsonl
    uv run python scripts/freeze_documents_v1.py --freeze evals/documents-v1  # needs runs/documents-v1-work/adjudications.jsonl

Rules (PLAN_27b, B2 revised). Train: a question keeps its native label only if both teachers chose it; otherwise the
question is dropped (a record with no question left is dropped). Development and test: a question is verified if all three
judges chose the native label; every other question goes to the adjudication queue and is adjudicated twice,
independently (runs/documents-v1-work/adjudication/out and out2); it is decided only where both agree, otherwise dropped.
Unparsed judge answers count as disagreement.
"""
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import PRIVATE_DATASET, SERVING_CONTEXT, read_json, read_jsonl, read_manifest, write_json, write_jsonl  # noqa: E402

WORK = Path("runs/documents-v1-work")
TEACHERS = ("deepseek/deepseek-v3.2", "alibaba/qwen3-235b-a22b-thinking")
JUDGES = ("anthropic/claude-opus-4.5", "openai/gpt-5", "google/gemini-3-flash")


def load(split):
    return read_jsonl(WORK / "candidates" / f"{split}.jsonl")


def answers(split, models):
    out = {}
    for m in models:
        path = WORK / "labels" / split / (m.replace("/", "__") + ".jsonl")
        out[m] = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
    return out


def vote(ans, model, rid, qid):
    r = ans[model].get(rid)
    a = (r or {}).get("answers") or {}
    return (a.get(qid) or {}).get("label"), (a.get(qid) or {}).get("reason", "")


def strip(q):
    return dict(q)   # kept whole: kev.data.materialize needs each question's "src"


def train_split():
    if not (WORK / "candidates" / "train.jsonl").exists(): return None, {}   # a held-out-only suite (documents-v2)
    recs, ans, kept, c = load("train"), answers("train", TEACHERS), [], Counter()
    for r in recs:
        qs = {}
        for qid, q in r["questions"].items():
            votes = [vote(ans, m, r["_meta"]["id"], qid)[0] for m in TEACHERS]
            c["questions"] += 1; c["unlabelled"] += any(v is None for v in votes)
            if all(v == q["label"] for v in votes): qs[qid] = strip(q); c["kept"] += 1
        if qs: kept.append({**r, "questions": qs})
    return kept, dict(c)


def eval_split(split, adjudications):
    if not (WORK / "candidates" / f"{split}.jsonl").exists(): return None, [], {}
    recs, ans, kept, queue, c = load(split), answers(split, JUDGES), [], [], Counter()
    for r in recs:
        qs, rid = {}, r["_meta"]["id"]
        for qid, q in r["questions"].items():
            votes = {m: vote(ans, m, rid, qid) for m in JUDGES}
            c["questions"] += 1
            if all(v[0] == q["label"] for v in votes.values()):
                qs[qid] = strip(q); c["verified_unanimous"] += 1; continue
            item = f"{rid}#{qid}"
            queue.append({"id": item, "split": split, "document": r["state"], "source": "cfpb", "question": {"type": "choice", "instructions": q["instructions"], "options": q["criteria"]},
                          "proposed_label": q["label"], "label_origin": "native", "judges": [{"model": m, "label": v[0], "rationale": v[1]} for m, v in votes.items()], "adjudication": None})
            adj = adjudications.get(item)
            if adj is None: c["awaiting_adjudication"] += 1; continue
            if adj["verdict"] == "drop": c["dropped_agreed" if adj.get("agreed") else "dropped_disagreement"] += 1; continue
            label = q["label"] if adj["verdict"] == "accept" else adj["label"]
            assert label in q["criteria"], item
            qs[qid] = {**strip(q), "label": label}; c["kept_native" if adj["verdict"] == "accept" else "relabelled"] += 1
        if qs: kept.append({**r, "questions": qs})
    return kept, queue, dict(c)


def read_verdicts(directory):
    out = {}
    for path in sorted(Path(directory).glob("shard-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if r.get("verdict") in ("accept", "relabel", "drop"): out[r["id"]] = r
    return out


def combine():
    """Agreement of two independent adjudications decides an item; any disagreement (or a missing verdict) drops it."""
    queue = [r["id"] for r in read_jsonl(WORK / "adjudication_queue.jsonl")]
    a, b = read_verdicts(WORK / "adjudication" / "out"), read_verdicts(WORK / "adjudication" / "out2")
    rows, c = [], Counter()
    for item in queue:
        x, y = a.get(item), b.get(item)
        key = lambda v: (v["verdict"], v.get("label") if v["verdict"] == "relabel" else None)
        if x and y and key(x) == key(y):
            rows.append({"id": item, "verdict": x["verdict"], "label": x.get("label") if x["verdict"] == "relabel" else None, "reasons": [x.get("reason", ""), y.get("reason", "")], "agreed": True}); c["agreed_" + x["verdict"]] += 1
        else:
            rows.append({"id": item, "verdict": "drop", "label": None, "reasons": [(x or {}).get("reason", "missing"), (y or {}).get("reason", "missing")], "agreed": False,
                         "verdicts": [(x or {}).get("verdict"), (y or {}).get("verdict")]}); c["disagreed_dropped" if x and y else "missing_dropped"] += 1
    write_jsonl(WORK / "adjudications.jsonl", rows)
    decided = sum(v for k, v in c.items() if k.startswith("agreed"))
    print(dict(c), f"adjudicator agreement {decided}/{len(queue)} = {decided / len(queue):.3f}")
    return dict(c)


def spot_check(test):
    """The protocol's human sample: 50 test questions drawn with a fixed seed, in the tools/review input format."""
    items = [(r, qid) for r in test for qid in r["questions"]]
    spot = random.Random("documents-v1-spot-check").sample(items, 50)
    write_jsonl(WORK / "spot_check.jsonl", [{"id": f"{r['_meta']['id']}#{qid}", "document": r["state"], "source": f"cfpb · {r['_meta']['length_bucket']} · {r['_meta']['chars']} chars",
                                              "question": {"type": "choice", "instructions": r["questions"][qid]["instructions"], "options": r["questions"][qid]["criteria"]},
                                              "proposed_label": r["questions"][qid]["label"], "label_origin": "frozen", "judges": [], "adjudication": None} for r, qid in spot])
    print(f"spot-check sample: 50 of {len(items)} test questions -> {WORK / 'spot_check.jsonl'}")


def upload_private(out, names):
    """Push the frozen partitions to the private mirror at the suite's path under evals/; returns the commit to pin."""
    from huggingface_hub import CommitOperationAdd, HfApi
    rel = out.resolve().relative_to(next(p for p in out.resolve().parents if p.name == "evals"))
    info = HfApi().create_commit(PRIVATE_DATASET, repo_type="dataset", commit_message=f"{rel}: frozen partitions",
                                 operations=[CommitOperationAdd(f"{rel}/{n}", str(out / n)) for n in names])
    return info.oid


def main():
    global WORK
    ap = argparse.ArgumentParser(); ap.add_argument("--report", action="store_true"); ap.add_argument("--freeze", default=""); ap.add_argument("--combine", action="store_true")
    ap.add_argument("--spot-check", action="store_true", help="write the 50-item human sample from the (final) test split")
    ap.add_argument("--work", default=str(WORK)); ap.add_argument("--version", default="documents-v1")
    ap.add_argument("--private", action="store_true", help="partitions go only to kev.suite.PRIVATE_DATASET; the manifest pins that commit")
    a = ap.parse_args()
    WORK = Path(a.work)
    if a.combine: return combine()
    adj_path = WORK / "adjudications.jsonl"
    adjudications = {r["id"]: r for r in read_jsonl(adj_path)} if adj_path.exists() else {}
    train, tc = train_split()
    splits, report, queue = {"train": train}, {"train": tc}, []
    for split in ("development", "test"):
        splits[split], q, report[split] = eval_split(split, adjudications); queue += q
    splits = {k: v for k, v in splits.items() if v is not None}; report = {k: v for k, v in report.items() if k in splits}
    print(json.dumps(report, indent=1))
    if a.spot_check:
        if report["test"].get("awaiting_adjudication"): raise SystemExit("test adjudications missing")
        return spot_check(splits["test"])
    write_jsonl(WORK / "adjudication_queue.jsonl", queue)
    print(f"adjudication queue: {len(queue)} items -> {WORK / 'adjudication_queue.jsonl'}")
    if not a.freeze: return
    if any(r.get("awaiting_adjudication") for r in report.values()): raise SystemExit("adjudications missing; nothing frozen")
    out = Path(a.freeze)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    files = {}
    for split, recs in splits.items():
        write_jsonl(out / f"{split}.jsonl", recs)
        files[f"{split}.jsonl"] = {"sha256": hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest(), "records": len(recs), "questions": sum(len(r["questions"]) for r in recs),
                                   "by_length": dict(Counter(r["_meta"]["length_bucket"] for r in recs))}
    spot_check(splits["test"])
    reviews = read_jsonl(WORK / "spot_check_reviews.jsonl")
    sample = {r["id"]: r["proposed_label"] for r in read_jsonl(WORK / "spot_check.jsonl")}
    if {r["id"] for r in reviews} != set(sample) or any(r["proposed_label"] != sample[r["id"]] for r in reviews): raise SystemExit("spot-check reviews do not match the sample")
    agreed = sum(r["verdict"] == "accept" for r in reviews)
    if agreed < 47: raise SystemExit(f"spot check {agreed}/50 is below the registered 47/50; not frozen")
    human = {"reviewer": "Jared Palmer", "tool": "tools/review", "sample": "50 test questions, seed documents-v1-spot-check", "agreement": f"{agreed}/50",
             "disagreements": [{"id": r["id"], "frozen": r["proposed_label"], "reviewer": r["label"], "verdict": r["verdict"]} for r in reviews if r["verdict"] != "accept"],
             "reviews_sha256": hashlib.sha256((WORK / "spot_check_reviews.jsonl").read_bytes()).hexdigest()}
    build = read_json(WORK / "candidates" / "build.json")
    mirror = {"mirror": {"dataset": PRIVATE_DATASET, "revision": upload_private(out, list(files))}} if a.private else {}
    write_json(out / "manifest.json", {"version": a.version, "partitions": list(splits), "locked": ["test"], "files": files, **mirror,
                                       "source": build["repo"] + "@" + build["revision"], "rights": "consumer narratives published by the US CFPB with consent; the CFPB considers them public domain for FOIA purposes" if a.private else "US CFPB consumer complaint database, US government work (public domain)",
                                       "label_protocol": "PLAN_27b B2 revised: train = native label kept where both open-weight teachers agree; development/test = native label verified by a unanimous three-judge panel or adjudicated",
                                       "teachers": TEACHERS, "judges": JUDGES, "adjudicators": "two independent Devin subagents per item (Claude family); decided only on agreement", "label_report": report, "spot_check": human,
                                       "description": f"AI-adjudicated, human spot-checked ({human['agreement']} agreement)",
                                       "trainable_sources": ["cfpb"], "holdout_sources": [], "context": SERVING_CONTEXT,
                                       "base_revisions": read_manifest("evals/v7/decision-v7")["base_revisions"], "candidates_build": build})
    print(f"frozen {out}; spot-check sample -> {WORK / 'spot_check.jsonl'}")


if __name__ == "__main__":
    main()
