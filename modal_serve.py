"""Serve Kev on Modal through vLLM (kev.vllm_model), and measure it against the torch serving path on the same GPU.

    uv run modal deploy modal_serve.py                                             # System One endpoint, vLLM backend
    uv run modal run modal_serve.py::parity --run jaredpalmer/kev-4b               # kev.benchmark: fp32 torch vs bf16 torch vs vLLM
    uv run modal run modal_serve.py::loadtest --run jaredpalmer/kev-4b             # p50/p99 + throughput at 1/8/32 concurrent clients

The two backends need two images: the torch path runs the locked kev environment (torch 2.8 + flash-linear-attention, as
skills/kev-deploy and modal_app.py do), vLLM brings its own torch, so its image installs vllm and kev's other
dependencies next to the local `kev/` source. Both mount the `kev-hf-cache` volume, which also keeps the merged vLLM
export (kev.vllm_model.export_dir) so only the first cold start merges. parity writes kev.benchmark result directories to
the `kev-runs` volume under /serve/<name> and pulls them to runs/serve/<name>; compare them with kev.compare.
Settings: KEV_MODEL (endpoint checkpoint, default jaredpalmer/kev-4b), KEV_GPU (default L40S), KEV_API_KEY (bearer auth,
uploaded as a Modal secret), KEV_APP_NAME (default kev-vllm).
"""
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
SETTINGS = {"KEV_MODEL": "jaredpalmer/kev-4b", "KEV_APP_NAME": "kev-vllm", "KEV_GPU": "L40S"}
SETTINGS = {k: os.environ.get(k, v) for k, v in SETTINGS.items()}
GPU = SETTINGS["KEV_GPU"]
SECRET = {k: os.environ[k] for k in ("KEV_API_KEY", "HF_TOKEN") if os.environ.get(k)}
ENV = {"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
       "TRITON_CACHE_DIR": "/hf/triton-cache", **SETTINGS}

app = modal.App(SETTINGS["KEV_APP_NAME"])
torch_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(ROOT), groups=[], extras=["serve"])
    .uv_pip_install("flash-linear-attention", "triton>=3.7.1")   # Gated DeltaNet kernels (fla needs triton >= 3.7.1 on Hopper)
    .env({**ENV, "KEV_BACKEND": "torch"})
    .add_local_python_source("kev")
    .add_local_dir(ROOT / "evals", "/root/evals")
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("vllm==0.30.0", "transformers>=5.17,<6", "peft>=0.21", "accelerate>=1.15.0", "datasets>=3.0",
                    "scikit-learn>=1.9.1", "fastapi>=0.115", "typesafe-sdk>=0.6.0", "uvicorn>=0.30",
                    "flash-linear-attention")   # fla: the torch fp32 reference in parity runs in this image too
    .env({**ENV, "KEV_BACKEND": "vllm"})
    .add_local_python_source("kev")
    .add_local_dir(ROOT / "evals", "/root/evals")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_dict(SECRET)] if SECRET else []
VOLUMES = {"/hf": hf_cache, "/runs": runs_volume}
WARMUP = {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "model": "kev-latest",
          "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                       "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}},
                        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"}}}


def load_server(run):
    """kev.serve's Server for `run`, loaded the way kev.serve.main does on CUDA (bf16, CUDA graphs for torch, KEV_BACKEND from the image)."""
    import torch
    from kev.api import SystemOneRequest
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.serve import Server
    ck = Checkpoint(run)
    tok, model = ck.load("cuda", replace(LoadOptions.from_env(), dtype=torch.bfloat16, cuda_graphs=True))
    server = Server(ck, tok, model, "cuda")
    server.answer(SystemOneRequest.model_validate(WARMUP))   # compile kernels / capture graphs before the first real request
    hf_cache.commit()
    return server


@app.cls(image=vllm_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes={"/hf": hf_cache}, secrets=secrets,
         scaledown_window=300, timeout=600, startup_timeout=1800)
@modal.concurrent(max_inputs=64)   # the engine batches concurrent requests; kev.serve no longer serializes them
class Kev:
    @modal.enter()
    def load(self):
        from kev.serve import app as api
        started = time.time()
        api.state.server = load_server(SETTINGS["KEV_MODEL"])
        print(f"serving {SETTINGS['KEV_MODEL']} via vllm, ready in {time.time() - started:.0f}s", flush=True)
        self.api = api

    @modal.asgi_app(label=f"{SETTINGS['KEV_APP_NAME']}-api")
    def web(self):
        return self.api


def benchmark(run, suite, name, dtype):
    """kev.benchmark on the suite's development partition with this image's backend at KEV_DTYPE=dtype, to /runs/serve/<name>."""
    out = Path("/runs/serve") / name
    try:
        subprocess.run([sys.executable, "-m", "kev.benchmark", "--run", run, "--suite", f"/root/{suite}", "--out", str(out), "--device", "cuda"],
                       check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root", "KEV_DTYPE": dtype})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return (out / "report.json").read_text(encoding="utf-8")


@app.function(image=torch_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def benchmark_torch(run, suite, name, dtype):
    return benchmark(run, suite, name, dtype)


@app.function(image=vllm_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def benchmark_vllm(run, suite, name):
    return benchmark(run, suite, name, "bf16")


def load_test(run, suite, levels, requests):
    """Server.probs from `c` client threads at once for each concurrency level c: per-request latency p50/p99 and
    requests/s over `requests` development records (the same sample for every level and backend)."""
    import random
    import statistics
    from concurrent.futures import ThreadPoolExecutor
    from kev.data import materialize
    from kev.suite import load_split
    server = load_server(run)
    records = [materialize(r) for r in load_split(f"/root/{suite}", "development")]
    sample = random.Random(0).choices(records, k=requests)

    def one(rec):
        t = time.perf_counter(); server.probs(rec); return time.perf_counter() - t

    for rec in sample: one(rec)   # warm every shape before timing: torch captures each new CUDA-graph bucket in the background
    while server.capture_lock.locked(): time.sleep(0.1)
    report = {"backend": server.model.backend, "gpu": GPU, "run": run, "levels": {}}
    for c in levels:
        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter()
            lat = sorted(pool.map(one, sample))
            wall = time.perf_counter() - start
        report["levels"][c] = {"p50_ms": round(1000 * statistics.median(lat), 1), "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1),
                               "requests_per_s": round(len(lat) / wall, 2), "questions_per_s": round(sum(len(r["questions"]) for r in sample) / wall, 2)}
        print(report["levels"][c], flush=True)
    return report


@app.function(image=torch_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def load_test_torch(run, suite, levels, requests):
    return load_test(run, suite, levels, requests)


@app.function(image=vllm_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def load_test_vllm(run, suite, levels, requests):
    return load_test(run, suite, levels, requests)


@app.local_entrypoint()
def parity(run: str = SETTINGS["KEV_MODEL"], suite: str = "evals/v7/decision-v7", tag: str = ""):
    """Three kev.benchmark runs of `run` on the same GPU type: torch fp32 (the reported-numbers path), torch bf16 (the
    current serving path) and vLLM bf16. Pulled to runs/serve/; `kev.compare --candidate runs/serve/<x> --reference ...`."""
    name = tag or run.split("/")[-1].replace("@", "-")
    jobs = [benchmark_torch.spawn(run, suite, f"{name}-torch-fp32", "fp32"), benchmark_torch.spawn(run, suite, f"{name}-torch-bf16", "bf16"),
            benchmark_vllm.spawn(run, suite, f"{name}-vllm-bf16")]
    for job in jobs: job.get()
    Path("runs/serve").mkdir(parents=True, exist_ok=True)
    for kind in ("torch-fp32", "torch-bf16", "vllm-bf16"):
        subprocess.run(["modal", "volume", "get", "--force", "kev-runs", f"/serve/{name}-{kind}", "runs/serve/"], check=True)
    print(f"pulled runs/serve/{name}-{{torch-fp32,torch-bf16,vllm-bf16}}")


@app.local_entrypoint()
def loadtest(run: str = SETTINGS["KEV_MODEL"], suite: str = "evals/v7/decision-v7", levels: str = "1,8,32", requests: int = 256):
    """Same records, same GPU type, both backends; prints one JSON report per backend."""
    import json
    cs = [int(c) for c in levels.split(",")]
    for report in (load_test_torch.remote(run, suite, cs, requests), load_test_vllm.remote(run, suite, cs, requests)):
        print(json.dumps(report, indent=1))
