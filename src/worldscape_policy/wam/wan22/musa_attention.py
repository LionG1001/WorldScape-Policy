"""MUSA FlashAttention-2 dispatch with an explicit, mask-preserving SDPA fallback."""
import functools
import logging
import os

import torch
from torch.nn import functional as F

from .sdpa import attention_lengths, variable_length_sdpa

LOGGER = logging.getLogger(__name__)
_OBSERVED = set()


@functools.lru_cache(maxsize=1)
def _flash_varlen():
    try:
        from flash_attn import flash_attn_varlen_func
        from flash_attn.backends import musa
    except ImportError:
        return None
    return flash_attn_varlen_func if musa.is_available() else None


def _record(path, reason):
    key = (path, reason)
    if key not in _OBSERVED:
        LOGGER.info("MUSA attention path=%s reason=%s", path, reason)
        _OBSERVED.add(key)


def musa_attention(q, k, v, *, q_lens=None, k_lens=None, dropout_p=0.,
                   softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), deterministic=False,
                   dtype=torch.bfloat16):
    backend = os.environ.get('WSP_MUSA_ATTENTION_BACKEND', 'sdpa').lower()
    if backend not in {'auto', 'flash_attn', 'sdpa'}:
        raise ValueError('WSP_MUSA_ATTENTION_BACKEND must be auto, flash_attn, or sdpa')
    # Checked against the muDNN-backed FA2 interface, not CUDA capability numbers.
    q_sizes = attention_lengths(q_lens, q.shape[0], q.shape[1])
    k_sizes = attention_lengths(k_lens, k.shape[0], k.shape[1])
    eligible = (q.device.type == 'musa' and 64 <= q.shape[-1] <= 192
                and q.shape[-1] == k.shape[-1] == v.shape[-1]
                and dropout_p == 0 and (window_size or (-1, -1)) == (-1, -1)
                and all(q_sizes) and all(k_sizes)
                and (not causal or q_sizes == k_sizes))
    func = _flash_varlen() if backend != 'sdpa' and eligible else None
    if backend == 'flash_attn' and eligible and func is None:
        raise RuntimeError('MUSA flash_attn requested but its MUSA extension is unavailable')
    if func is None:
        reason = 'explicit' if backend == 'sdpa' else ('unsupported contract' if not eligible else 'extension unavailable')
        _record('sdpa', reason)
        return variable_length_sdpa(q, k, v, q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale,
            causal=causal, window_size=window_size, dtype=dtype)

    output_dtype = q.dtype
    target_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else dtype
    def pack(x, sizes):
        x = x.to(target_dtype)
        if all(n == x.shape[1] for n in sizes):
            return x.flatten(0, 1)
        return torch.cat([row[:n] for row, n in zip(x, sizes, strict=True)])
    packed_q, packed_k, packed_v = pack(q, q_sizes), pack(k, k_sizes), pack(v, k_sizes)
    if q_scale is not None:
        packed_q = packed_q * q_scale
    def offsets(sizes):
        total, cumulative = 0, [0]
        for size in sizes:
            total += size
            cumulative.append(total)
        return torch.tensor(cumulative, dtype=torch.int32, device=q.device)
    # Do not catch runtime/OOM/backend correctness failures and silently reroute.
    out = func(packed_q, packed_k, packed_v, offsets(q_sizes), offsets(k_sizes),
               max(q_sizes), max(k_sizes), dropout_p=0., softmax_scale=softmax_scale,
               causal=causal, window_size=(-1, -1), deterministic=deterministic)
    _record('flash_attn', 'successful forward')
    if all(n == q.shape[1] for n in q_sizes):
        return out.unflatten(0, (q.shape[0], q.shape[1])).to(output_dtype)
    rows = out.split(q_sizes)
    return torch.stack([F.pad(row, (0, 0, 0, 0, 0, q.shape[1]-n))
                        for row, n in zip(rows, q_sizes, strict=True)]).to(output_dtype)
