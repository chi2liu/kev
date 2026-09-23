"""documents-v1: apply the B2 label rules to the candidates and the LLM answers (scripts/label_documents_v1.py), write the
adjudication queue and, once adjudications exist, the frozen suite.

    uv run python scripts/freeze_documents_v1.py --report                  # agreement tables, writes the adjudication queue
    uv run python scripts/freeze_documents_v1.py --freeze evals/documents-v1  # needs runs/documents-v1-work/adjudications.jsonl

Rules (PLAN_27b, B2 revised). Train: a question keeps its native label only if both teachers chose it; otherwise the
question is dropped (a record with no question left is dropped). Development and test: a question is verified if all three
judges chose the native label; every other question goes to the adjudication queue, and its adjudication decides it
(keep the native label, relabel with a reason, or drop). Unparsed judge answers count as disagreement.
"""
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import write_json, write_jsonl  # noqa: E402

WORK = Path("runs/documents-v1-work")
TEACHERS = ("deepseek/deepseek-v3.2", "alibaba/qwen3-235b-a22b-thinking")
JUDGES = ("anthropic/claude-opus-4.5", "openai/gpt-5", "google/gemini-3-flash")


def load(split):
    return [json.loads(l) for l in open(WORK / "candidates" / f"{split}.jsonl")]


def answers(split, models):
    out = {}
    for m in models:
        path = WORK / "labels" / split / (m.replace("/", "__") + ".jsonl")
        out[m] = {r["id"]: r for r in map(json.loads, open(path))} if path.exists() else {}
    return out


def vote(ans, model, rid, qid):
    r = ans[model].get(rid)
    a = (r or {}).get("answers") or {}
    return (a.get(qid) or {}).get("label"), (a.get(qid) or {}).get("reason", "")


def strip(q):
    return {k: v for k, v in q.items() if k != "src"}


def train_split():
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
            if adj["verdict"] == "drop": c["dropped"] += 1; continue
            label = q["label"] if adj["verdict"] == "accept" else adj["label"]
            assert label in q["criteria"], item
            qs[qid] = {**strip(q), "label": label}; c["kept_native" if adj["verdict"] == "accept" else "relabelled"] += 1
        if qs: kept.append({**r, "questions": qs})
    return kept, queue, dict(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", action="store_true"); ap.add_argument("--freeze", default="")
    a = ap.parse_args()
    adj_path = WORK / "adjudications.jsonl"
    adjudications = {r["id"]: r for r in map(json.loads, open(adj_path))} if adj_path.exists() else {}
    train, tc = train_split()
    splits, report, queue = {"train": train}, {"train": tc}, []
    for split in ("development", "test"):
        splits[split], q, report[split] = eval_split(split, adjudications); queue += q
    print(json.dumps(report, indent=1))
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
    rng = random.Random("documents-v1-spot-check")
    test_items = [(r, qid) for r in splits["test"] for qid in r["questions"]]
    spot = rng.sample(test_items, 50)
    write_jsonl(WORK / "spot_check.jsonl", [{"id": f"{r['_meta']['id']}#{qid}", "document": r["state"], "source": "cfpb",
                                              "question": {"type": "choice", "instructions": r["questions"][qid]["instructions"], "options": r["questions"][qid]["criteria"]},
                                              "proposed_label": r["questions"][qid]["label"], "label_origin": "frozen", "judges": [], "adjudication": None} for r, qid in spot])
    build = json.load(open(WORK / "candidates" / "build.json"))
    write_json(out / "manifest.json", {"version": "documents-v1", "partitions": ["train", "development", "test"], "locked": ["test"], "files": files,
                                       "source": build["repo"] + "@" + build["revision"], "rights": "US CFPB consumer complaint database, US government work (public domain)",
                                       "label_protocol": "PLAN_27b B2 revised: train = native label kept where both open-weight teachers agree; development/test = native label verified by a unanimous three-judge panel or adjudicated",
                                       "teachers": TEACHERS, "judges": JUDGES, "label_report": report, "spot_check": "pending (runs/documents-v1-work/spot_check.jsonl, 50 test items)",
                                       "trainable_sources": ["cfpb"], "candidates_build": build})
    print(f"frozen {out}; spot-check sample -> {WORK / 'spot_check.jsonl'}")


if __name__ == "__main__":
    main()
