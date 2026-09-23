"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth
when KEV_API_KEY is set (unset = open server, the local default). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison). KEV_PREFIX_CACHE / KEV_PREFIX_MIN_TOKENS size the state-prefix cache; KEV_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts). Backend and precision follow LoadOptions (KEV_BACKEND, KEV_DTYPE, ...): on Apple
Silicon the hybrid Qwen3.5 checkpoints run on MLX by default, elsewhere on torch in bf16.
"""
import argparse, atexit, hmac, os, queue, random, threading, time, uuid
from concurrent.futures import Future
import torch
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, is_hub_id
from .device import default_device, sync
from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PREFIX_MIN_TOKENS = os.environ.get("KEV_PREFIX_MIN_TOKENS")               # states shorter than this are not cached; default = the model's prefix_min_tokens (0 for hybrid backbones and MLX, 384 for attention-only torch models)
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
API_KEY = os.environ.get("KEV_API_KEY")                                  # unset = open server; set = require Authorization: Bearer <key>, as the TypeSafe clients always send
HOT_BUCKET = 3                                                           # eager passes after which a busy server captures a bucket's CUDA graph anyway
MAX_BATCH = 64                                                           # requests the model thread takes at once (kev.cuda_graphs splits them to fit its buffers)
MODEL_NAMES = ("kev-latest", "jev-latest")                               # both names serve this checkpoint; jev-latest is the TypeSafe SDK default model, so an unconfigured client works


@dataclass
class Server:
    """The loaded checkpoint, the state-prefix cache, and the one model thread that runs every forward pass.

    Request threads encode their record and queue it; the model thread takes everything queued when it becomes free and
    runs it as one batch (model.probs_batch: with CUDA graphs, one state pass for the batch's new states and one row pass
    for all its questions; otherwise one request at a time), then answers each request. When nothing is waiting it
    captures one pending CUDA graph. `lock` is held around each batch and capture: hold it to use the model directly."""
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    prefix_cache: dict = field(default_factory=dict)   # (state token ids, option_isolation) -> prefix, in LRU order
    prefix_hits: int = 0
    prefix_misses: int = 0
    batches: int = 0
    batched_requests: int = 0
    release_date: str = field(default="")   # for the TypeSafe model card; resolved once (may ask the Hub)

    def __post_init__(self):
        self.release_date = self.release_date or self.checkpoint.release_date()
        self.queue, self.busy, self.stopping = queue.Queue(), False, threading.Event()
        self.thread = threading.Thread(target=self._work, name="kev-model", daemon=True)
        self.thread.start()
        atexit.register(self.close)   # a daemon thread killed inside a CUDA call at interpreter exit aborts the process

    def close(self):
        """Stop the model thread after its current pass."""
        self.stopping.set(); self.thread.join(timeout=10)

    @property
    def prefix_min_tokens(self):
        return int(PREFIX_MIN_TOKENS) if PREFIX_MIN_TOKENS else self.model.prefix_min_tokens

    def probs(self, rec):
        """Probabilities for one record, from the model thread. The state prefix (tokens up to the first question) is
        cached across requests, so a repeated state only pays for its question rows. latency_ms is the model time of the
        batch the request ran in (not its wait in the queue)."""
        try: enc = self.model.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        except ValueError as e: raise HTTPException(422, str(e))
        done = Future()
        self.queue.put((enc, done))
        return done.result()

    def _work(self):
        graphs = getattr(self.model, "graphs", None)
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty:   # idle: capture one pending CUDA graph (a request arriving now waits for at most this one, ~0.4 s)
                if graphs is not None and graphs.pending: self._capture(graphs)
                continue
            while len(batch) < MAX_BATCH:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            self.busy = True
            try:
                with self.lock: self._run(batch)
            except Exception as e:   # answer every waiting request; the thread lives on
                for _, done in batch:
                    if not done.done(): done.set_exception(e)
            finally:
                self.busy = False
            if graphs is not None and graphs.pending and max(graphs.eager_runs.values()) >= HOT_BUCKET:
                self._capture(graphs)                              # busy, but this bucket keeps running eagerly

    def _capture(self, graphs):
        self.busy = True
        try:
            with self.lock: graphs.capture_pending(limit=1)
        finally:
            self.busy = False

    def _run(self, batch):
        cache, encs = self.prefix_cache, [enc for enc, _ in batch]
        states = [enc["seg"].count(0) for enc in encs]
        keys = [(tuple(enc["ids"][:Ls]), bool(enc.get("option_isolation"))) for enc, Ls in zip(encs, states)]
        cached = [bool(PREFIX_CACHE_SIZE) and Ls >= self.prefix_min_tokens for Ls in states]
        prefixes = [cache.get(k) if c else None for k, c in zip(keys, cached)]
        sync(self.device); t = time.time()
        if getattr(self.model, "graphs", None) is not None:
            ps, new = self.model.probs_batch(encs, prefixes)
        else:   # every other backend: one request at a time, as before batching
            ps, new = map(list, zip(*[(self.model.probs(e), None) if not use else (self.model.probs_with_prefix(e, p), p) if p is not None
                                      else self.model.probs_and_prefix(e) for e, p, use in zip(encs, prefixes, cached)]))
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.batches += 1; self.batched_requests += len(batch)
        for (enc, done), key, use, prefix, fresh, p, Ls in zip(batch, keys, cached, prefixes, new, ps, states):
            if use:
                cache.pop(key, None); cache[key] = fresh              # (re)insert = most recently used
                while len(cache) > PREFIX_CACHE_SIZE: cache.pop(next(iter(cache)))
                self.prefix_hits += prefix is not None; self.prefix_misses += prefix is None
            done.set_result(([q.tolist() for q in p], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": dt, "prefix_cache_hit": prefix is not None}))

    def wait_idle(self):
        """Block until nothing is queued or running and no CUDA graph waits to be captured (benchmarks, warm-up)."""
        graphs = getattr(self.model, "graphs", None)
        while not self.queue.empty() or self.busy or (graphs is not None and graphs.pending): time.sleep(0.01)

    def answer(self, req):
        """The /v1/systemone response body for one request."""
        rec, meta = to_record(prepare(req))
        ps, m = self.probs(rec)
        answers = to_answers(ps, meta)
        return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)}, "latency_ms": m["latency_ms"]}


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    return req.model_copy(update={"state": with_date_facts(req.state)}) if DATE_FACTS else req


app = FastAPI(title="kev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["x-typesafe-request-id"])


@app.middleware("http")
async def typesafe(request, call_next):
    """Bearer auth (when API_KEY is set) and the request id every TypeSafe client reads off the response."""
    if API_KEY and request.url.path.startswith("/v1") and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"}, 401, {"www-authenticate": "Bearer"})
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    return resp


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    return server().answer(req)


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)   # each order is a forward pass; 0 divided by nothing, unbounded counts ran forever (#30)
    seed: int = 0


@app.post("/v1/systemone/permute")
def systemone_permute(r: PermuteSystemOne):
    """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
    q = r.request.questions.get(r.question)
    if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
    rng = random.Random(r.seed); keys = list(q.criteria); runs = []
    for i in range(r.n_perm):
        order = list(keys)
        if i > 0: rng.shuffle(order)
        one = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        resp = server().answer(one); a = resp["answers"][r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": resp["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    parts = [server().answer(req.model_copy(update={"questions": {qid: q}})) for qid, q in req.questions.items()]
    answers = {qid: a for p in parts for qid, a in p["answers"].items()}
    return {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts), "output_tokens": output_tokens(server().tok, answers)},
            "latency_ms": round(sum(p["latency_ms"] for p in parts), 1)}


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus the Kev serving details
    a client may ignore: the run, the base, the device, the backend and precision, the temperature, prefix-cache stats."""
    s = server()
    ck, meta = s.checkpoint, s.checkpoint.meta
    card = {"description": f"Kev pointer head on {meta.base}, serving {ck.requested} at temperature {s.model.head.temperature:.2f}",
            "release_date": s.release_date,
            "run": ck.requested, "base": meta.base, "lora": meta.lora, "device": s.device, "backend": s.model.backend, "dtype": s.model.dtype,
            "temperature": s.model.head.temperature,
            "cuda_graphs": {"captured": graphs.captures, "kept": len(graphs.graphs)} if (graphs := getattr(s.model, "graphs", None)) else None,
            "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": s.prefix_min_tokens, "hits": s.prefix_hits,
                             "misses": s.prefix_misses, "cached_states": len(s.prefix_cache)},
            "batches": {"count": s.batches, "requests": s.batched_requests, "queued": s.queue.qsize()}}
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    dev = default_device()
    opts = LoadOptions.from_env()
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default: 2-4.5x faster than fp32 on an L4, same answers (LoadOptions.dtype); KEV_DTYPE=fp32 for the exact path
    if dev == "cuda" and opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)   # serving default: a pass is ~2,000 kernel launches, so replaying graphs cuts warm latency several-fold (kev.cuda_graphs); KEV_CUDA_GRAPHS=0 to decline
    if opts.backend is None: opts = replace(opts, backend="auto")   # serving default: MLX for the hybrid Qwen3.5 checkpoints on Apple Silicon (LoadOptions.backend); KEV_BACKEND=torch to decline
    ck = Checkpoint(run)
    tok, model = ck.load(dev, opts)
    app.state.server = Server(ck, tok, model, dev)
    print(f"serving {ck.requested} ({ck.path}) on {dev} via {model.backend} ({model.dtype}) :{a.port}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
