from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    import zstandard as zstd
except ImportError as exc:
    raise RuntimeError("zstandard is required for zstd level-22 compression.") from exc


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weights",
    "loop_embeddings",
    "loop_gates",
    "smear.gate",
    "bigram.scale",
)
FP16_KEEP_NAME_PATTERNS = ("tok_emb",)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 0.9999984


@dataclass(frozen=True)
class Hyperparameters:
    data_path: Path = Path(os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
    tokenizer_path: Path = Path(os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
    results_path: Path = Path("./results/phase2_result.json")
    artifact_path: Path = Path("./results/final_model_phase2.int8.ptz")

    seed: int = int(os.environ.get("SEED", "1337"))
    iterations: int = int(os.environ.get("ITERATIONS", "20000"))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", "50"))

    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "0"))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", "524288"))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", "2048"))

    val_eval_tokens: int = int(os.environ.get("VAL_EVAL_TOKENS", "524288"))
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", "64"))
    eval_batch_seqs: int = int(os.environ.get("EVAL_BATCH_SEQS", "8"))
    eval_docs: int = int(os.environ.get("EVAL_DOCS", "50"))
    cache_max_tokens: int = int(os.environ.get("CACHE_MAX_TOKENS", "4096"))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", "1024"))
    model_dim: int = int(os.environ.get("MODEL_DIM", "512"))
    num_unique_layers: int = int(os.environ.get("NUM_UNIQUE_LAYERS", "7"))
    num_loops: int = int(os.environ.get("NUM_LOOPS", "2"))
    num_heads: int = int(os.environ.get("NUM_HEADS", "8"))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", "4"))
    mlp_mult: int = int(os.environ.get("MLP_MULT", "3"))

    tie_embeddings: bool = True
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", "0.005"))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))
    rope_base: float = float(os.environ.get("ROPE_BASE", "10000.0"))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", "1.5"))

    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", "2048"))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", "128"))

    matrix_lr: float = float(os.environ.get("MATRIX_LR", "0.02"))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", "0.02"))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", "0.99"))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
    beta1: float = float(os.environ.get("BETA1", "0.9"))
    beta2: float = float(os.environ.get("BETA2", "0.95"))
    adam_eps: float = float(os.environ.get("ADAM_EPS", "1e-8"))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", "0.04"))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))

    ternary_qat_start_frac: float = float(os.environ.get("TERNARY_QAT_START_FRAC", "0.85"))
    ternary_lr_scale_on_enable: float = float(os.environ.get("TERNARY_LR_SCALE_ON_ENABLE", "0.5"))

    swa_start_frac: float = float(os.environ.get("SWA_START_FRAC", "0.4"))
    swa_every: int = int(os.environ.get("SWA_EVERY", "50"))

    zstd_level: int = int(os.environ.get("ZSTD_LEVEL", "22"))

    compile_training: bool = bool(int(os.environ.get("COMPILE_TRAIN_MODEL", "1")))
    compile_backend: str = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

    @property
    def total_depth(self) -> int:
        return self.num_unique_layers * self.num_loops

    @property
    def ternary_enable_step(self) -> int:
        return min(self.iterations, max(1, int(self.iterations * self.ternary_qat_start_frac)))

    @property
    def swa_start_step(self) -> int:
        return min(self.iterations, max(1, int(self.iterations * self.swa_start_frac)))


def resolve_path(root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (root / maybe_relative).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size}, got {file.stat().st_size}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int, max_tokens: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    target_tokens = max(max_tokens, seq_len + 1)
    chunks: list[Tensor] = []
    loaded = 0
    for file in files:
        shard = load_data_shard(file)
        chunks.append(shard)
        loaded += shard.numel()
        if loaded >= target_tokens:
            break
    tokens = torch.cat(chunks).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    usable = min(usable, max_tokens)
    return tokens[: usable + 1]


def extract_first_validation_docs(val_files: list[Path], bos_id: int, num_docs: int) -> list[Tensor]:
    docs: list[Tensor] = []
    current: list[int] = []
    for file in val_files:
        shard = load_data_shard(file).numpy()
        for tok in shard:
            token = int(tok)
            if token == bos_id:
                if current:
                    docs.append(torch.tensor(current, dtype=torch.int64))
                    if len(docs) >= num_docs:
                        return docs
                current = [token]
            elif not current:
                current = [token]
            else:
                current.append(token)
    if current and len(docs) < num_docs:
        docs.append(torch.tensor(current, dtype=torch.int64))
    if len(docs) < num_docs:
        raise ValueError(f"Only found {len(docs)} validation docs, need at least {num_docs}")
    return docs[:num_docs]


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class SingleGpuTokenLoader:
    def __init__(self, pattern: str, device: torch.device):
        self.stream = TokenStream(pattern)
        self.device = device

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        if batch_tokens % seq_len != 0:
            raise ValueError(f"batch_tokens must be divisible by seq_len, got {batch_tokens}/{seq_len}")
        local = self.stream.take(batch_tokens + 1).to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True)
        y = local[1:].reshape(-1, seq_len).to(self.device, non_blocking=True)
        return x, y


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class TernaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("_ternary_enabled", torch.tensor(0.0), persistent=False)

    def set_ternary_enabled(self, enabled: bool) -> None:
        self._ternary_enabled.fill_(1.0 if enabled else 0.0)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        alpha = weight.abs().mean(dim=1, keepdim=True)
        w_q = (weight / (alpha + 1e-8)).round().clamp(-1, 1)
        w_ste = weight + (w_q - weight).detach()
        ternary_weight = w_ste * alpha
        enabled = self._ternary_enabled.to(dtype=weight.dtype)
        effective_weight = weight + enabled * (ternary_weight - weight)
        return F.linear(x, effective_weight, self.bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.scale = 1.0 / math.sqrt(self.head_dim)
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    @staticmethod
    def build_causal_mask(seq_len: int, cache_len: int, device: torch.device) -> Tensor:
        right = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
        if cache_len == 0:
            return right
        left = torch.ones((seq_len, cache_len), device=device, dtype=torch.bool)
        return torch.cat((left, right), dim=1)

    def _expand_kv_for_gqa(self, t: Tensor) -> Tensor:
        if self.num_kv_heads == self.num_heads:
            return t
        repeat = self.num_heads // self.num_kv_heads
        return t.repeat_interleave(repeat, dim=1)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if kv_cache is None:
            full_k = k
            full_v = v
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        else:
            prev_k, prev_v = kv_cache
            cache_len = prev_k.size(2)
            prev_k = prev_k.to(dtype=k.dtype)
            prev_v = prev_v.to(dtype=v.dtype)
            # Cache order must be past-before-present for causal decoding.
            full_k = torch.cat((prev_k, k), dim=2)
            full_v = torch.cat((prev_v, v), dim=2)
            if cache_len > 0:
                if not torch.equal(full_k[:, :, :1, :], prev_k[:, :, :1, :]):
                    raise RuntimeError("Invalid cache concatenation order for K (expected cached before current).")
                if not torch.equal(full_v[:, :, :1, :], prev_v[:, :, :1, :]):
                    raise RuntimeError("Invalid cache concatenation order for V (expected cached before current).")
            full_k_exp = self._expand_kv_for_gqa(full_k)
            full_v_exp = self._expand_kv_for_gqa(full_v)
            mask = self.build_causal_mask(seqlen, cache_len, x.device)
            if cache_len > 0 and not bool(mask[:, :cache_len].all().item()):
                raise RuntimeError("Cached tokens are masked out; cached prefix must be fully visible.")
            attn_scores = torch.matmul(q, full_k_exp.transpose(-2, -1)) * self.scale
            attn_scores = attn_scores.masked_fill(~mask[None, None, :, :], torch.finfo(attn_scores.dtype).min)
            attn = torch.softmax(attn_scores, dim=-1)
            y = torch.matmul(attn, full_v_exp)

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        out = self.proj(y)

        new_cache: tuple[Tensor, Tensor] | None = None
        if cache_max_tokens is not None:
            keep = min(cache_max_tokens, full_k.size(2))
            new_cache = (
                full_k[:, :, -keep:, :].detach(),
                full_v[:, :, -keep:, :].detach(),
            )
        return out, new_cache


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = TernaryLinear(dim, hidden, bias=False)
        self.proj = TernaryLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, token_ids: Tensor) -> Tensor:
        t = token_ids.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, cache_max_tokens=cache_max_tokens)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x, new_cache


def run_unet_once(
    x: Tensor,
    x0: Tensor,
    blocks: nn.ModuleList,
    num_encoder_layers: int,
    num_decoder_layers: int,
    skip_weights: Tensor,
    kv_cache_blocks: list[tuple[Tensor, Tensor] | None] | None = None,
    cache_max_tokens: int | None = None,
    use_gradient_checkpointing: bool = False,
) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None] | None]:
    if kv_cache_blocks is not None and len(kv_cache_blocks) != len(blocks):
        raise ValueError(f"kv_cache_blocks length {len(kv_cache_blocks)} does not match blocks length {len(blocks)}")
    use_ckpt = use_gradient_checkpointing and torch.is_grad_enabled() and cache_max_tokens is None and kv_cache_blocks is None
    skips: list[Tensor] = []
    new_cache: list[tuple[Tensor, Tensor] | None] | None = [] if cache_max_tokens is not None else None
    for i in range(num_encoder_layers):
        if use_ckpt:
            block = blocks[i]

            def run_block(inp: Tensor, resid: Tensor, _block: nn.Module = block) -> Tensor:
                out, _ = _block(inp, resid, kv_cache=None, cache_max_tokens=None)
                return out

            x = checkpoint(run_block, x, x0, use_reentrant=False)
            layer_new_cache = None
        else:
            layer_cache = None if kv_cache_blocks is None else kv_cache_blocks[i]
            x, layer_new_cache = blocks[i](x, x0, kv_cache=layer_cache, cache_max_tokens=cache_max_tokens)
        skips.append(x)
        if new_cache is not None:
            new_cache.append(layer_new_cache)
    for i in range(num_decoder_layers):
        if skips and i < skip_weights.size(0):
            x = x + skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        layer_idx = num_encoder_layers + i
        if use_ckpt:
            block = blocks[layer_idx]

            def run_block(inp: Tensor, resid: Tensor, _block: nn.Module = block) -> Tensor:
                out, _ = _block(inp, resid, kv_cache=None, cache_max_tokens=None)
                return out

            x = checkpoint(run_block, x, x0, use_reentrant=False)
            layer_new_cache = None
        else:
            layer_cache = None if kv_cache_blocks is None else kv_cache_blocks[layer_idx]
            x, layer_new_cache = blocks[layer_idx](x, x0, kv_cache=layer_cache, cache_max_tokens=cache_max_tokens)
        if new_cache is not None:
            new_cache.append(layer_new_cache)
    return x, new_cache


class Phase2GPT(nn.Module):
    def __init__(self, cfg: Hyperparameters):
        super().__init__()
        if cfg.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {cfg.logit_softcap}")
        self.cfg = cfg
        self.num_unique_layers = cfg.num_unique_layers
        self.num_loops = cfg.num_loops
        self.num_encoder_layers = cfg.num_unique_layers // 2
        self.num_decoder_layers = cfg.num_unique_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.total_effective_layers = cfg.num_unique_layers * cfg.num_loops

        self.logit_softcap = cfg.logit_softcap
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.bigram = BigramHashEmbedding(cfg.bigram_vocab_size, cfg.bigram_dim, cfg.model_dim)
        self.smear = SmearGate(cfg.model_dim)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, cfg.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.model_dim,
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    cfg.mlp_mult,
                    cfg.rope_base,
                    cfg.qk_gain_init,
                )
                for _ in range(cfg.num_unique_layers)
            ]
        )
        self.loop_embeddings = nn.Parameter(torch.zeros(cfg.num_loops, cfg.model_dim, dtype=torch.float32))
        self.loop_gates = nn.Parameter(torch.ones(cfg.num_loops, dtype=torch.float32))
        self.use_gradient_checkpointing = True
        self.final_norm = RMSNorm()
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.cfg.tied_embed_init_std)
            self.tok_emb.weight.data = self.tok_emb.weight.data.to(torch.float16)
        num_layers = max(self.cfg.total_depth, 1)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def forward_hidden(
        self,
        input_ids: Tensor,
        kv_cache: list[tuple[Tensor, Tensor] | None] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None] | None]:
        if self.training and (kv_cache is not None or cache_max_tokens is not None):
            raise RuntimeError("Neural cache is evaluation-only and must not be used during training.")
        if kv_cache is not None and len(kv_cache) != self.total_effective_layers:
            raise ValueError(
                f"kv_cache length {len(kv_cache)} does not match effective layers {self.total_effective_layers}"
            )
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids).to(dtype=x.dtype)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        new_cache_flat: list[tuple[Tensor, Tensor] | None] | None = [] if cache_max_tokens is not None else None

        for loop_idx in range(self.num_loops):
            x = x + self.loop_embeddings[loop_idx].to(dtype=x.dtype)[None, None, :]
            loop_input = x
            if cache_max_tokens is not None:
                loop_cache = None
                if kv_cache is not None:
                    start = loop_idx * self.num_unique_layers
                    end = start + self.num_unique_layers
                    loop_cache = kv_cache[start:end]
            else:
                loop_cache = None
            x, loop_new_cache = run_unet_once(
                x,
                x0,
                self.blocks,
                self.num_encoder_layers,
                self.num_decoder_layers,
                self.skip_weights,
                kv_cache_blocks=loop_cache,
                cache_max_tokens=cache_max_tokens,
                use_gradient_checkpointing=(self.use_gradient_checkpointing and self.training),
            )
            gate = self.loop_gates[loop_idx].to(dtype=x.dtype)
            x = loop_input + gate * (x - loop_input)
            if new_cache_flat is not None and loop_new_cache is not None:
                new_cache_flat.extend(loop_new_cache)

        return self.final_norm(x), new_cache_flat

    def forward_logits(
        self,
        input_ids: Tensor,
        kv_cache: list[tuple[Tensor, Tensor] | None] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None] | None]:
        x, new_cache = self.forward_hidden(input_ids, kv_cache=kv_cache, cache_max_tokens=cache_max_tokens)
        logits_proj = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits, new_cache

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits, _ = self.forward_logits(input_ids, kv_cache=None, cache_max_tokens=None)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )


def enable_ternary(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.set_ternary_enabled(True)
            count += 1
    return count


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        n = b * m + c * m @ m
        x = a * x + n @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if p.ndim != 2:
                    raise ValueError(f"Muon expects matrix params only; got {tuple(p.shape)}")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                if weight_decay > 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                p.add_(g.to(dtype=p.dtype), alpha=-lr)
        return loss


def split_optimizer_parameters(model: nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_control = any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        if p.ndim == 2 and not is_control:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    if not matrix_params:
        raise RuntimeError("No matrix parameters found for Muon.")
    if not scalar_params:
        raise RuntimeError("No scalar/vector parameters found for AdamW.")
    return matrix_params, scalar_params


def compute_bpb(loss_sum: float, token_count: float, byte_count: float) -> float:
    if token_count <= 0 or byte_count <= 0:
        raise ValueError(f"Invalid counters: token_count={token_count}, byte_count={byte_count}")
    val_loss = loss_sum / token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count / byte_count
    return bits_per_token * tokens_per_byte


@torch.no_grad()
def eval_val_sliding(
    model: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    seq_len: int,
    stride: int,
    batch_seqs: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    model.eval()
    for bi in range(0, len(window_starts), batch_seqs):
        batch_ws = window_starts[bi : bi + batch_seqs]
        bsz = len(batch_ws)
        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens: list[int] = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            chunk = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64)
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits, _ = model.forward_logits(x_batch, kv_cache=None, cache_max_tokens=None)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y_batch.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            score_start = 0 if ws == 0 else max(wlen - stride, 0)
            scored_nll = nll[i, score_start:wlen].to(torch.float64)
            loss_sum += float(scored_nll.sum().item())
            token_count += float(wlen - score_start)
            tgt = y_batch[i, score_start:wlen]
            prev = x_batch[i, score_start:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += float(tb.sum().item())
    val_loss = loss_sum / token_count
    return val_loss, compute_bpb(loss_sum, token_count, byte_count)


@torch.no_grad()
def eval_docs_neural_cache(
    model: Phase2GPT,
    docs: list[Tensor],
    device: torch.device,
    seq_len: int,
    stride: int,
    cache_max_tokens: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    *,
    use_cache: bool,
    debug_cache_stats: bool = False,
) -> tuple[float, float, int, int]:
    model.eval()
    if model.training:
        raise RuntimeError("Neural cache evaluation requires model.eval().")
    cache_active = use_cache and cache_max_tokens > 0
    loss_sum = 0.0
    token_count = 0
    byte_count = 0.0
    windows = 0
    debug_printed = False
    for doc in docs:
        if doc.numel() <= 1:
            continue
        x_doc = doc[:-1]
        y_doc = doc[1:]
        layer_cache: list[tuple[Tensor, Tensor] | None] | None = (
            [None for _ in range(model.total_effective_layers)] if cache_active else None
        )
        for ws in range(0, x_doc.numel(), stride):
            end = min(ws + seq_len, x_doc.numel())
            wlen = end - ws
            if wlen < 1:
                continue
            x_win = x_doc[ws:end].to(device=device, dtype=torch.int64).unsqueeze(0)
            y_win = y_doc[ws:end].to(device=device, dtype=torch.int64).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if cache_active:
                    logits, layer_cache = model.forward_logits(
                        x_win,
                        kv_cache=layer_cache,
                        cache_max_tokens=cache_max_tokens,
                    )
                    if debug_cache_stats and not debug_printed and layer_cache is not None:
                        for layer_state in layer_cache:
                            if layer_state is None:
                                continue
                            cached_k = layer_state[0].to(torch.float32)
                            k_mean = float(cached_k.mean().item())
                            k_std = float(cached_k.std(unbiased=False).item())
                            print(
                                f"neural_cache_debug:first_cached_k mean={k_mean:.6e} std={k_std:.6e}",
                                flush=True,
                            )
                            if abs(k_mean) < 1e-8 or k_std < 1e-8 or k_std > 1e3:
                                print(
                                    "neural_cache_debug:WARNING cached K stats look suspicious (near-zero or too large)",
                                    flush=True,
                                )
                            debug_printed = True
                            break
                else:
                    logits, _ = model.forward_logits(x_win, kv_cache=None, cache_max_tokens=None)
            score_start = 0 if ws == 0 else max(wlen - stride, 0)
            scored_logits = logits[:, score_start:, :].reshape(-1, logits.size(-1)).float()
            scored_targets = y_win[:, score_start:].reshape(-1)
            nll_sum = F.cross_entropy(scored_logits, scored_targets, reduction="sum")
            loss_sum += float(nll_sum.item())
            token_count += int(scored_targets.numel())

            prev_ids = x_win[:, score_start:].reshape(-1)
            tgt_ids = scored_targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += float(token_bytes.to(torch.float64).sum().item())
            windows += 1
    if debug_cache_stats and cache_active and not debug_printed:
        print("neural_cache_debug:WARNING no cached K tensor was observed.", flush=True)
    val_loss = loss_sum / float(token_count)
    return val_loss, compute_bpb(loss_sum, float(token_count), byte_count), windows, token_count


def _classify_param(name: str) -> str:
    if "tok_emb" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 31.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]) -> tuple[dict[str, Tensor], dict[str, object]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(dtype=torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(
    result: dict[str, Tensor],
    meta: dict[str, object],
    template_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta[name]
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def update_swa_state(
    model: nn.Module,
    swa_state: dict[str, Tensor] | None,
    swa_count: int,
) -> tuple[dict[str, Tensor], int]:
    state = model.state_dict()
    if swa_state is None:
        new_state: dict[str, Tensor] = {}
        for name, tensor in state.items():
            if tensor.is_floating_point():
                new_state[name] = tensor.detach().cpu().float()
            else:
                new_state[name] = tensor.detach().cpu().clone()
        return new_state, 1
    for name, tensor in state.items():
        if tensor.is_floating_point():
            swa_state[name].add_(tensor.detach().cpu().float())
        else:
            swa_state[name] = tensor.detach().cpu().clone()
    return swa_state, swa_count + 1


def apply_swa_state(model: nn.Module, swa_state: dict[str, Tensor], swa_count: int) -> None:
    current_state = model.state_dict()
    avg_state: dict[str, Tensor] = {}
    for name, tensor in current_state.items():
        if tensor.is_floating_point():
            avg_state[name] = (swa_state[name] / float(swa_count)).to(dtype=tensor.dtype, device=tensor.device)
        else:
            avg_state[name] = swa_state[name].to(device=tensor.device)
    model.load_state_dict(avg_state, strict=True)


def main() -> None:
    args = Hyperparameters()
    NEURAL_CACHE_ENABLED = os.environ.get("NEURAL_CACHE_ENABLED", "0") == "1"
    root = Path(__file__).resolve().parent
    dataset_dir = resolve_path(root, args.data_path)
    tokenizer_path = resolve_path(root, args.tokenizer_path)
    results_path = resolve_path(root, args.results_path)
    artifact_path = resolve_path(root, args.artifact_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    print("Single-GPU launch command: torchrun --standalone --nproc_per_node=1 experiment_phase2_combined.py", flush=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("This script supports single GPU mode only. Use --nproc_per_node=1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    if args.train_batch_tokens % args.train_seq_len != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS must be divisible by TRAIN_SEQ_LEN, got {args.train_batch_tokens}/{args.train_seq_len}"
        )
    if args.eval_stride <= 0 or args.eval_stride > args.train_seq_len:
        raise ValueError(f"EVAL_STRIDE must be in [1, TRAIN_SEQ_LEN], got {args.eval_stride}")

    train_glob = str(dataset_dir / "fineweb_train_*.bin")
    val_glob = str(dataset_dir / "fineweb_val_*.bin")
    train_files = sorted(glob.glob(train_glob))
    val_files = sorted(glob.glob(val_glob))
    if not train_files:
        raise FileNotFoundError(f"No train shards found for pattern: {train_glob}")
    if not val_files:
        raise FileNotFoundError(f"No val shards found for pattern: {val_glob}")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer has no BOS id.")

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(val_glob, seq_len=args.train_seq_len, max_tokens=args.val_eval_tokens)
    val_docs = extract_first_validation_docs([Path(p) for p in val_files], bos_id=bos_id, num_docs=args.eval_docs)
    train_loader = SingleGpuTokenLoader(train_glob, device=device)

    compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    NO_COMPILE = os.environ.get("NO_COMPILE", "0") == "1"

    base_model = Phase2GPT(args).to(device)
    training_model: nn.Module = base_model
    if NO_COMPILE or not args.compile_training:
        training_model = base_model
        if NO_COMPILE:
            print("compile_disabled:NO_COMPILE=1", flush=True)
        else:
            print("compile_disabled:COMPILE_TRAIN_MODEL=0", flush=True)
    else:
        training_model = torch.compile(
            base_model,
            dynamic=False,
            fullgraph=False,
            backend=compile_backend,
        )
        print(f"compile_backend:{compile_backend}", flush=True)
    print(f"neural_cache_enabled:{int(NEURAL_CACHE_ENABLED)}", flush=True)

    matrix_params, scalar_params = split_optimizer_parameters(base_model)
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.weight_decay,
    )
    adamw_kwargs = dict(
        lr=args.scalar_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if torch.cuda.is_available():
        adamw_kwargs["fused"] = True
    optimizer_adamw = torch.optim.AdamW(scalar_params, **adamw_kwargs)
    optimizers = [optimizer_muon, optimizer_adamw]

    print(
        "config:"
        f" model_dim={args.model_dim}"
        f" num_unique_layers={args.num_unique_layers}"
        f" num_loops={args.num_loops}"
        f" total_depth={args.total_depth}"
        f" num_heads={args.num_heads}"
        f" num_kv_heads={args.num_kv_heads}"
        f" mlp_mult={args.mlp_mult}"
        f" bigram_vocab_size={args.bigram_vocab_size}"
        f" train_batch_tokens={args.train_batch_tokens}"
        f" train_seq_len={args.train_seq_len}",
        flush=True,
    )
    print(
        "schedule:"
        f" ternary_enable_step={args.ternary_enable_step}/{args.iterations}"
        f" ternary_frac={args.ternary_qat_start_frac:.2f}"
        f" lr_scale_on_ternary={args.ternary_lr_scale_on_enable}"
        f" swa_start_step={args.swa_start_step}/{args.iterations}"
        f" swa_every={args.swa_every}",
        flush=True,
    )
    print(
        "optimizer:"
        f" Muon(matrix) lr={args.matrix_lr} wd={args.weight_decay}"
        f" AdamW(scalars) lr={args.scalar_lr} wd={args.weight_decay}",
        flush=True,
    )
    print(f"loop_gates_init={base_model.loop_gates.detach().cpu().tolist()}", flush=True)

    ternary_layers_enabled = 0
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    train_steps_completed = 0
    t0 = time.perf_counter()

    for step in range(1, args.iterations + 1):
        elapsed = time.perf_counter() - t0
        if args.max_wallclock_seconds > 0 and elapsed >= args.max_wallclock_seconds:
            print(f"Stopping at step {step - 1} due to max wallclock {args.max_wallclock_seconds}s", flush=True)
            break

        if step == args.ternary_enable_step:
            ternary_layers_enabled = enable_ternary(training_model)
            if ternary_layers_enabled == 0 and training_model is not base_model:
                ternary_layers_enabled = enable_ternary(base_model)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] *= args.ternary_lr_scale_on_enable
            print(
                f"step:{step} ternary_enabled=True ternary_layers:{ternary_layers_enabled}"
                f" lr_muon:{optimizer_muon.param_groups[0]['lr']:.6g}"
                f" lr_adamw:{optimizer_adamw.param_groups[0]['lr']:.6g}",
                flush=True,
            )

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = training_model(x, y)
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite training loss at step {step}: {loss_value}")
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer_muon.step()
        optimizer_adamw.step()
        train_steps_completed = step

        if step == 1 or step % args.train_log_every == 0:
            print(
                f"step:{step}/{args.iterations}"
                f" train_loss:{loss_value:.6f}"
                f" elapsed_s:{(time.perf_counter() - t0):.2f}",
                flush=True,
            )

        if step >= args.swa_start_step and (step - args.swa_start_step) % args.swa_every == 0:
            swa_state, swa_count = update_swa_state(base_model, swa_state, swa_count)
            print(f"step:{step} swa_update count:{swa_count}", flush=True)

    if train_steps_completed == 0:
        raise RuntimeError("No training steps completed.")

    if swa_state is not None and swa_count > 1:
        print(f"Applying SWA average over {swa_count} checkpoints", flush=True)
        apply_swa_state(base_model, swa_state, swa_count)

    torch.cuda.synchronize()
    pre_quant_val_loss, pre_quant_val_bpb = eval_val_sliding(
        model=base_model,
        val_tokens=val_tokens,
        device=device,
        seq_len=args.train_seq_len,
        stride=args.eval_stride,
        batch_seqs=args.eval_batch_seqs,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )

    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstd.ZstdCompressor(level=args.zstd_level).compress(quant_raw)
    artifact_path.write_bytes(quant_blob)
    artifact_size_bytes = int(artifact_path.stat().st_size)

    decompressed = zstd.ZstdDecompressor().decompress(artifact_path.read_bytes())
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)

    post_quant_model = Phase2GPT(args).to(device)
    post_quant_model.load_state_dict(deq_state, strict=True)
    post_quant_model.eval()
    torch.cuda.synchronize()
    post_quant_val_loss, post_quant_val_bpb = eval_val_sliding(
        model=post_quant_model,
        val_tokens=val_tokens,
        device=device,
        seq_len=args.train_seq_len,
        stride=args.eval_stride,
        batch_seqs=args.eval_batch_seqs,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )

    # Fresh non-compiled model for neural cache eval.
    neural_cache_model = Phase2GPT(args).to(device)
    neural_cache_model.load_state_dict(deq_state, strict=True)
    neural_cache_model.eval()
    torch.cuda.synchronize()
    cache_sweep_sizes = (0, 256, 1024)
    cache_sweep: dict[str, dict[str, float | int]] = {}
    for cache_size in cache_sweep_sizes:
        sweep_use_cache = NEURAL_CACHE_ENABLED and cache_size > 0
        sweep_cache_max = cache_size if sweep_use_cache else 0
        sweep_loss, sweep_bpb, sweep_windows, sweep_tokens = eval_docs_neural_cache(
            model=neural_cache_model,
            docs=val_docs,
            device=device,
            seq_len=args.train_seq_len,
            stride=args.eval_stride,
            cache_max_tokens=sweep_cache_max,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            use_cache=sweep_use_cache,
            debug_cache_stats=(cache_size == 256 and sweep_use_cache),
        )
        cache_sweep[str(cache_size)] = {
            "use_cache": int(sweep_use_cache),
            "effective_cache_max_tokens": int(sweep_cache_max),
            "val_loss": sweep_loss,
            "val_bpb": sweep_bpb,
            "windows": sweep_windows,
            "scored_tokens": sweep_tokens,
        }
        print(
            f"neural_cache_sweep cache_size={cache_size} effective_cache={sweep_cache_max} val_bpb:{sweep_bpb:.6f}",
            flush=True,
        )

    effective_post_cache_size = args.cache_max_tokens if NEURAL_CACHE_ENABLED else 0
    effective_post_use_cache = NEURAL_CACHE_ENABLED and effective_post_cache_size > 0
    if str(effective_post_cache_size) in cache_sweep:
        post_cache_val_loss = float(cache_sweep[str(effective_post_cache_size)]["val_loss"])
        post_cache_val_bpb = float(cache_sweep[str(effective_post_cache_size)]["val_bpb"])
        cache_windows = int(cache_sweep[str(effective_post_cache_size)]["windows"])
        cache_scored_tokens = int(cache_sweep[str(effective_post_cache_size)]["scored_tokens"])
    else:
        post_cache_val_loss, post_cache_val_bpb, cache_windows, cache_scored_tokens = eval_docs_neural_cache(
            model=neural_cache_model,
            docs=val_docs,
            device=device,
            seq_len=args.train_seq_len,
            stride=args.eval_stride,
            cache_max_tokens=effective_post_cache_size,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            use_cache=effective_post_use_cache,
            debug_cache_stats=effective_post_use_cache,
        )

    print(f"val_bpb pre-quant: {pre_quant_val_bpb:.6f}", flush=True)
    print(f"val_bpb post-quant: {post_quant_val_bpb:.6f}", flush=True)
    print(f"val_bpb post-neural-cache: {post_cache_val_bpb:.6f}", flush=True)
    print(f"artifact_size_bytes: {artifact_size_bytes}", flush=True)

    result_payload = {
        "status": "DONE",
        "run_mode": "single_gpu",
        "launch_command": "torchrun --standalone --nproc_per_node=1 experiment_phase2_combined.py",
        "config": {
            **asdict(args),
            "data_path": str(dataset_dir),
            "tokenizer_path": str(tokenizer_path),
            "results_path": str(results_path),
            "artifact_path": str(artifact_path),
            "neural_cache_enabled": NEURAL_CACHE_ENABLED,
        },
        "schedule": {
            "iterations_target": args.iterations,
            "steps_completed": train_steps_completed,
            "ternary_qat_start_frac": args.ternary_qat_start_frac,
            "ternary_enable_step": args.ternary_enable_step,
            "ternary_lr_scale_on_enable": args.ternary_lr_scale_on_enable,
            "ternary_layers_enabled": ternary_layers_enabled,
            "swa_start_frac": args.swa_start_frac,
            "swa_start_step": args.swa_start_step,
            "swa_every": args.swa_every,
            "swa_count": swa_count,
        },
        "architecture": {
            "model_dim": args.model_dim,
            "num_unique_layers": args.num_unique_layers,
            "num_loops": args.num_loops,
            "total_depth": args.total_depth,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "mlp_mult": args.mlp_mult,
            "bigram_vocab_size": args.bigram_vocab_size,
            "smear_gate": True,
            "fp16_tied_embeddings": True,
            "loop_gates_init": "ones",
            "loop_embeddings_init": "zeros",
            "cache_max_tokens": args.cache_max_tokens,
        },
        "metrics": {
            "pre_quant": {
                "val_loss": pre_quant_val_loss,
                "val_bpb": pre_quant_val_bpb,
            },
            "post_quant": {
                "val_loss": post_quant_val_loss,
                "val_bpb": post_quant_val_bpb,
            },
            "post_neural_cache": {
                "val_loss": post_cache_val_loss,
                "val_bpb": post_cache_val_bpb,
                "windows": cache_windows,
                "scored_tokens": cache_scored_tokens,
                "effective_cache_max_tokens": effective_post_cache_size,
                "use_cache": int(effective_post_use_cache),
            },
            "neural_cache_sweep": cache_sweep,
            "artifact_size_bytes": artifact_size_bytes,
        },
    }
    results_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
