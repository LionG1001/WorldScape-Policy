"""Portable SDPA with FlashAttention's bottom-right mask convention."""

import torch
from torch.nn import functional as F


def variable_length_sdpa(
    q, k, v, *, q_lens=None, k_lens=None, dropout_p=0.0,
    softmax_scale=None, q_scale=None, causal=False,
    window_size=(-1, -1), dtype=torch.bfloat16,
):
    output_dtype = q.dtype
    cast = lambda x: x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
    q, k, v = (cast(x).transpose(1, 2) for x in (q, k, v))
    if q_scale is not None:
        q = q * q_scale
    left, right = window_size or (-1, -1)
    options = dict(dropout_p=dropout_p, scale=softmax_scale,
                   enable_gqa=q.shape[1] != k.shape[1])
    if q_lens is None and k_lens is None and not causal and left < 0 and right < 0:
        return F.scaled_dot_product_attention(q, k, v, **options).transpose(1, 2).to(output_dtype).contiguous()
    batch, _, padded_q, _ = q.shape
    q_sizes = [padded_q] * batch if q_lens is None else q_lens.tolist()
    k_sizes = [k.shape[2]] * batch if k_lens is None else k_lens.tolist()
    outputs = []
    for i, (lq, lk) in enumerate(zip(q_sizes, k_sizes, strict=True)):
        lq, lk = int(lq), int(lk)
        if not 0 <= lq <= padded_q or not 0 <= lk <= k.shape[2]:
            raise ValueError("attention lengths exceed padded sequence dimensions")
        if lq == 0 or lk == 0:
            out = q[i:i+1, :, :lq, :1].expand(-1, -1, -1, v.shape[-1]) * 0
        else:
            mask = None
            if causal or left >= 0 or right >= 0:
                # FlashAttention aligns unequal causal sequences at the right.
                row = torch.arange(lq, device=q.device)[:, None] + lk - lq
                col = torch.arange(lk, device=q.device)[None, :]
                mask = torch.ones((lq, lk), dtype=torch.bool, device=q.device)
                if causal:
                    mask &= col <= row
                if left >= 0:
                    mask &= col >= row - left
                if right >= 0:
                    mask &= col <= row + right
            out = F.scaled_dot_product_attention(
                q[i:i+1, :, :lq], k[i:i+1, :, :lk], v[i:i+1, :, :lk],
                attn_mask=mask, **options,
            )
        outputs.append(F.pad(out, (0, 0, 0, padded_q - lq)))
    return torch.cat(outputs).transpose(1, 2).to(output_dtype).contiguous()
