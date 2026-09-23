"""documents-v2 candidates: a held-out test set only (PLAN.md round 7, confirmation 5), drawn exactly as documents-v1
(scripts/build_documents_v1.py: same source snapshot, products, issues and length buckets) from the narratives v1 never
drew: every v1 candidate (all three splits, dropped questions included) is excluded by text hash and complaint id. Its
partitions go only to the private mirror (kev.suite.PRIVATE_DATASET); never commit them.

    uv run python scripts/build_documents_v2.py --out runs/documents-v2-work/candidates
"""
import argparse, hashlib, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_documents_v1 import BUCKETS, ISSUE_OF, ISSUES, PRODUCTS, RAW, REPO, REVISION, rows, text_key  # noqa: E402
from kev.suite import write_json, write_jsonl  # noqa: E402

PER_CELL = 22
V1 = Path("runs/documents-v1-work/candidates")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", default="documents-v2")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    v1 = [json.loads(l) for split in ("train", "development", "test") for l in open(V1 / f"{split}.jsonl", encoding="utf-8")]
    seen, v1_ids = {r["_meta"]["text_sha256"] for r in v1}, {r["_meta"]["id"] for r in v1}
    cells, stats = defaultdict(list), Counter()
    for r in rows():
        text, key = (r["complaint_what_happened"] or "").strip(), RAW.get(r["product"])
        if not text or key is None: continue
        bucket = next((b for b, lo, hi in BUCKETS if lo <= len(text) < hi), None)
        if bucket is None: continue
        h = text_key(text)
        if h in seen or f"cfpb/{r['complaint_id']}" in v1_ids: stats["excluded_or_duplicate"] += 1; continue
        seen.add(h); cells[key, bucket].append({**r, "text": text, "key": key, "bucket": bucket, "text_sha256": h})
    rng, recs = random.Random(a.seed), []
    for (key, bucket), pool in sorted(cells.items()):
        rng.shuffle(pool)
        for r in pool[:PER_CELL]:
            qs = {"product": {"type": "choice", "instructions": "Which kind of financial product is this complaint about?",
                              "criteria": {k: d for k, (d, _) in PRODUCTS.items()}, "label": key, "src": "cfpb_product"}}
            issue = ISSUE_OF.get((key, r["issue"]))
            if issue:
                qs["issue"] = {"type": "choice", "instructions": "What is the main problem the consumer describes?",
                               "criteria": {k: raws[0] for k, raws in ISSUES[key].items()}, "label": issue, "src": f"cfpb_issue_{key}"}
            recs.append({"state": r["text"], "questions": qs,
                         "_meta": {"id": f"cfpb/{r['complaint_id']}", "source": "cfpb", "group_id": f"cfpb/{r['text_sha256'][:20]}", "variant": "clean",
                                   "length_bucket": bucket, "chars": len(r["text"]), "company": r["company"], "date_received": r["date_received"],
                                   "product_raw": r["product"], "issue_raw": r["issue"], "sub_issue_raw": r["sub_issue"], "text_sha256": r["text_sha256"],
                                   "provenance": {"source": "CFPB consumer complaint database", "via": f"hf:{REPO}@{REVISION}",
                                                  "rights": "consumer narratives published by the CFPB with consent; the CFPB considers them public domain for FOIA purposes"}}})
    rng.shuffle(recs)
    out.mkdir(parents=True); write_jsonl(out / "test.jsonl", recs)
    write_json(out / "build.json", {"repo": REPO, "revision": REVISION, "seed": a.seed, "per_cell": PER_CELL, "excluded_v1_candidates": len(v1), "stats": dict(stats),
                                    "per_split": {"test": len(recs)}, "questions": {"test": sum(len(r["questions"]) for r in recs)}, "buckets": BUCKETS,
                                    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(len(recs), "records,", sum(len(r["questions"]) for r in recs), "questions", dict(stats))


if __name__ == "__main__":
    main()
