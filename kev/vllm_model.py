"""CUDA serving backend: vLLM runs the backbone, Kev keeps its encoder and pointer head.

The torch path answers one request at a time under kev.serve's lock, so concurrent clients queue behind each other and
the GPU runs one small prefill per step. Here every question row (state + branch, `rows_of`, the same row form the torch
path uses for the hybrid Qwen3.5 backbones) is one vLLM pooling request: the engine batches rows across concurrent
requests (continuous batching over a paged KV cache, fused Gated DeltaNet kernels, CUDA graphs) and returns each row's
final hidden states ("token_embed" with ALL pooling, no activation). The readout is the very same fp32 `PointerHead` (with
the checkpoint's temperature) the torch and MLX paths apply, on the `<decide>` and `</opt>` positions.

vLLM cannot load a PEFT adapter onto every projection Kev trains (the DeltaNet in_proj_* are packed differently), so the
adapter is folded into the base in fp32 by the torch loader (the exact merge LoadOptions.merge describes), rounded once to
the serving dtype and exported as a plain `*ForCausalLM` checkpoint. The export is cached under $HF_HOME/kev-vllm, keyed
by everything that changes its weights, so only the first cold start pays for it.

vLLM is not a kev dependency (it pins its own torch); it lives in the Modal serving image (modal_serve.py). Selected by
`LoadOptions(backend="vllm")` (KEV_BACKEND=vllm) in `kev.checkpoint`; parity against the fp32 torch path and the load test
against torch bf16 are `modal run modal_serve.py::parity` / `::loadtest`.
"""
import asyncio, hashlib, json, os, shutil, threading, uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm import PoolingParams, TokensPrompt
from vllm.config import PoolerConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .model import SERVE_MAX_PACKED, PointerHead, encode, rows_of

EXPORT_VERSION = 1   # bump when the export layout changes, so cached exports are rebuilt
GPU_MEMORY_UTILIZATION = float(os.environ.get("KEV_VLLM_GPU_MEMORY", "0.85"))   # share of the GPU vLLM may take (weights + KV / state cache)
MAX_NUM_SEQS = int(os.environ.get("KEV_VLLM_MAX_SEQS", "256"))                   # rows the engine schedules at once


def export_dir(checkpoint, dtype, lora_scale):
    """Where the merged export of this checkpoint lives: a hash of the adapter, the base and every merge setting."""
    h = hashlib.sha256(checkpoint.file("adapter_model.safetensors").read_bytes())
    h.update(json.dumps([EXPORT_VERSION, checkpoint.meta.base, checkpoint.meta.base_revision, str(dtype), lora_scale]).encode())
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "kev-vllm"
    return root / h.hexdigest()[:16]


def export_merged(lm, tok, out, dtype):
    """Write a merged backbone (the text model DecisionModel.lm) as a checkpoint vLLM loads as `*ForCausalLM`. Kev never
    reads the vocab head, so the export ties it to the embeddings instead of storing one. Atomic: written beside `out`,
    then renamed, so a crashed export is never mistaken for a finished one."""
    tmp = out.with_name(out.name + f".tmp-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    cfg = lm.config
    cfg.architectures = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[cfg.model_type]]
    cfg.tie_word_embeddings = True
    cfg.dtype = dtype
    cfg.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    save_file({f"model.{k}": v.to(dtype).contiguous() for k, v in lm.state_dict().items()}, str(tmp / "model.safetensors"))
    try:
        tmp.rename(out)
    except OSError:          # another container finished the same export first
        shutil.rmtree(tmp)
    return out


class VLLMDecisionModel:
    """Prefill-only scorer: hidden states from a vLLM engine, logits from the shared torch PointerHead."""
    backend, device, hybrid, option_isolation = "vllm", "vllm", True, False
    concurrent = True        # kev.serve skips its lock and its state-prefix cache: every row goes to the engine whole
    prefix_min_tokens = None

    def __init__(self, model_dir, dtype, head_dim=256):
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        self._dtype = str(dtype).removeprefix("torch.")
        self.head = PointerHead(cfg["hidden_size"], dp=head_dim).eval()
        args = AsyncEngineArgs(model=str(model_dir), runner="pooling", dtype=self._dtype, max_model_len=SERVE_MAX_PACKED,
                               pooler_config=PoolerConfig(tok_pooling_type="ALL"), enable_prefix_caching=False,
                               gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_num_seqs=MAX_NUM_SEQS)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="kev-vllm", daemon=True).start()
        self.engine = self._run(self._start(args))

    async def _start(self, args):
        return AsyncLLM.from_engine_args(args)   # inside the loop: its output handler is a task on this loop

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    @property
    def dtype(self):
        return self._dtype

    def eval(self):
        self.head.eval(); return self

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=False, **kw)

    async def _hidden(self, ids):
        """[L, d] final hidden states of one causal row."""
        params = PoolingParams(task="token_embed", use_activation=False)
        async for out in self.engine.encode(TokensPrompt(prompt_token_ids=ids), params, uuid.uuid4().hex):
            last = out
        return last.outputs.data

    async def _logits(self, enc):
        S, _, rows = rows_of(enc)
        hs = await asyncio.gather(*(self._hidden(S + r["ids"]) for r in rows))
        out = []
        with torch.no_grad():
            for h, r in zip(hs, rows):
                picked = h[torch.tensor([len(S) + r["decide"], *(len(S) + o for o in r["opts"])])].float().cpu()
                out.append(self.head(picked[0], picked[1:]))
        return out

    def forward(self, enc):
        """List of logits tensors, one per question. Thread-safe: callers on many threads share the engine."""
        return self._run(self._logits(enc))

    def probs(self, enc):
        return [F.softmax(z, -1) for z in self.forward(enc)]

    def probs_and_prefix(self, enc):
        return self.probs(enc), None

    def probs_with_prefix(self, enc, prefix):
        return self.probs(enc)

    def shutdown(self):
        self.engine.shutdown()
