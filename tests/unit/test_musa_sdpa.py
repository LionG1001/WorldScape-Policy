import pytest
import torch
from worldscape_policy.wam.wan22.sdpa import variable_length_sdpa


@pytest.mark.parametrize('causal,window', [(False, (-1, -1)), (True, (-1, -1)), (False, (1, 1))])
def test_lengths_scale_mask_and_gradients(causal, window):
    torch.manual_seed(17)
    q = torch.randn(2, 5, 2, 8, requires_grad=True)
    k = torch.randn(2, 6, 2, 8, requires_grad=True)
    v = torch.randn(2, 6, 2, 8, requires_grad=True)
    ql, kl = [5, 3], [3, 6]
    actual = variable_length_sdpa(q, k, v, q_lens=torch.tensor(ql),
        k_lens=torch.tensor(kl), causal=causal, window_size=window,
        softmax_scale=0.2, q_scale=0.7, dtype=torch.float32)
    expected = []
    for i, (a, b) in enumerate(zip(ql, kl)):
        qi, ki, vi = (x[i].transpose(0, 1) for x in (q, k, v))
        scores = (qi[:, :a] * 0.7) @ ki[:, :b].transpose(-1, -2) * 0.2
        row = torch.arange(a)[:, None] + b - a
        col = torch.arange(b)[None, :]
        mask = torch.ones(a, b, dtype=torch.bool)
        if causal:
            mask &= col <= row
        if window[0] >= 0:
            mask &= col >= row-window[0]
        if window[1] >= 0:
            mask &= col <= row+window[1]
        # Avoid NaN gradients on fully masked rows in this explicit reference.
        any_key = mask.any(-1, keepdim=True)
        scores = scores.masked_fill(~mask, -float('inf'))
        scores = torch.where(any_key, scores, 0)
        probs = scores.softmax(-1) * any_key
        out = (probs @ vi[:, :b]).transpose(0, 1)
        expected.append(torch.nn.functional.pad(out, (0, 0, 0, 0, 0, 5-a)))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    grads = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    refs = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for grad, ref in zip(grads, refs):
        torch.testing.assert_close(grad, ref, atol=2e-6, rtol=1e-5)
