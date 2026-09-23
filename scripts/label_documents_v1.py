"""Label documents-v1 candidates with LLMs through the Vercel AI Gateway (PLAN_27b B2, revised protocol), blind to the
native labels. One call per (model, document) answers all of that document's questions; answers are cached per model and
split, so a rerun resumes. A shared spend ledger enforces a hard cap across every run.

    export AI_GATEWAY_API_KEY=$(cat ~/.config/kev/ai_gateway_key)
    uv run python scripts/label_documents_v1.py --split train --models deepseek/deepseek-v3.2,alibaba/qwen3-235b-a22b-thinking
    uv run python scripts/label_documents_v1.py --split test --models anthropic/claude-opus-4.5,openai/gpt-5,google/gemini-3-flash

Roles (the protocol): training labels come from two open-weight teachers and are kept only where both agree with the native
label; development and test labels are the native label checked by a three-judge panel from other model families. No Jev.
"""
import argparse, json, os, re, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORK = Path("runs/documents-v1-work")
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
SYSTEM = ("You label consumer financial complaints. For each question choose exactly one option key, using only what the "
          "complaint says. If several options fit, choose the one that best describes the consumer's main problem.")
lock = threading.Lock()


def prompt(rec):
    lines = [f"Complaint:\n<<<\n{rec['state']}\n>>>\n", "Questions:"]
    for qid, q in rec["questions"].items():
        lines.append(f"[{qid}] {q['instructions']}")
        lines += [f"  - {k}: {d}" for k, d in q["criteria"].items()]
    shape = ", ".join(f'"{qid}": {{"label": "<option key>", "reason": "<one short sentence>"}}' for qid in rec["questions"])
    lines.append(f"\nAnswer with only a JSON object: {{{shape}}}")
    return "\n".join(lines)


def parse(text, rec):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group())
    except json.JSONDecodeError: return None
    out = {}
    for qid, q in rec["questions"].items():
        a = obj.get(qid)
        label = a.get("label") if isinstance(a, dict) else a
        out[qid] = {"label": label if label in q["criteria"] else None, "reason": (a.get("reason", "") if isinstance(a, dict) else "")[:300]}
    return out


def call(model, rec, key, retries=6):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(rec)}],
            "max_tokens": 6000 if "thinking" in model or model.startswith(("openai/gpt-5", "google/gemini")) else 800}
    if model.startswith("openai/gpt-5"): body["reasoning_effort"] = "low"
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data, {"authorization": f"Bearer {key}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r: resp = json.load(r)
            text = resp["choices"][0]["message"].get("content") or ""
            return {"answers": parse(text, rec), "cost": float((resp.get("usage") or {}).get("cost") or 0), "raw": text[:2000]}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1: time.sleep(2 ** attempt + 1); continue
            return {"answers": None, "cost": 0.0, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            if attempt < retries - 1: time.sleep(2 ** attempt + 1); continue
            return {"answers": None, "cost": 0.0, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "development", "test"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--cap", type=float, default=80.0, help="hard cap in dollars across every run (the ledger at runs/documents-v1-work/labels/spend.json)")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N documents (a smoke run)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    key = os.environ["AI_GATEWAY_API_KEY"]
    if "jev" in a.models.lower(): raise SystemExit("no Jev in any role")
    recs = [json.loads(l) for l in open(WORK / "candidates" / f"{a.split}.jsonl")]
    if a.limit: recs = recs[:a.limit]
    ledger_path = WORK / "labels" / "spend.json"; ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = json.load(open(ledger_path)) if ledger_path.exists() else {"total": 0.0, "by_model": {}}
    for model in a.models.split(","):
        out = WORK / "labels" / a.split / (model.replace("/", "__") + ".jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        todo = [r for r in recs if r["_meta"]["id"] not in done]
        print(f"{model} on {a.split}: {len(done)} cached, {len(todo)} to label; spend so far ${ledger['total']:.2f} of ${a.cap:.2f}", flush=True)
        stop, n, bad = threading.Event(), 0, 0
        with open(out, "a") as f, ThreadPoolExecutor(a.workers) as pool:
            futures = {pool.submit(lambda r=r: None if stop.is_set() else call(model, r, key)): r for r in todo}
            for fut in as_completed(futures):
                res, rec = fut.result(), futures[fut]
                if res is None: continue
                with lock:
                    ledger["total"] += res["cost"]; ledger["by_model"][model] = ledger["by_model"].get(model, 0.0) + res["cost"]
                    f.write(json.dumps({"id": rec["_meta"]["id"], "model": model, **res}) + "\n"); f.flush()
                    n += 1; bad += res["answers"] is None
                    if n % 100 == 0:
                        json.dump(ledger, open(ledger_path, "w"), indent=1)
                        print(f"  {n}/{len(todo)} (unparsed {bad}) ${ledger['total']:.2f}", flush=True)
                    if ledger["total"] >= a.cap and not stop.is_set():
                        stop.set(); print(f"  spend cap ${a.cap:.2f} reached; stopping", flush=True)
        json.dump(ledger, open(ledger_path, "w"), indent=1)
        print(f"  done: {n} labelled, {bad} unparsed; ledger ${ledger['total']:.2f}", flush=True)
        if stop.is_set(): break


if __name__ == "__main__":
    main()
