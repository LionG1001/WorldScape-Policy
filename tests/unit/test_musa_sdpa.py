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


def test_empty_keys_keep_zero_gradients_for_all_inputs():
    q = torch.randn(2, 3, 2, 8, requires_grad=True)
    k = torch.randn(2, 4, 2, 8, requires_grad=True)
    v = torch.randn(2, 4, 2, 8, requires_grad=True)
    out = variable_length_sdpa(q, k, v, q_lens=torch.tensor([3, 0]),
                              k_lens=torch.tensor([0, 0]), dtype=torch.float32)
    assert torch.count_nonzero(out) == 0
    for grad in torch.autograd.grad(out.sum(), (q, k, v)):
        assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize('lengths', [[1], [1.5, 2], [-1, 2], [4, 2], [[1], [2]]])
def test_reject_invalid_lengths(lengths):
    q = torch.randn(2, 3, 2, 8)
    with pytest.raises(ValueError):
        variable_length_sdpa(q, q, q, q_lens=torch.tensor(lengths), dtype=torch.float32)


def test_explicit_torch_backend_preserves_mask_and_scaling(monkeypatch):
    from worldscape_policy.wam.wan22.lctx_attention import AttentionModule
    monkeypatch.delenv('ATTENTION_BACKEND', raising=False)
    module = AttentionModule(2, 8, backend='torch', q_scale=0.7,
                             softmax_scale=0.2, causal=True, dtype=torch.float32)
    assert module.backend == 'torch'
    q, k, v = torch.randn(2, 5, 2, 8), torch.randn(2, 6, 2, 8), torch.randn(2, 6, 2, 8)
    ql, kl = torch.tensor([5, 3]), torch.tensor([3, 6])
    actual = module(q, k, v, ql, kl)
    expected = variable_length_sdpa(q, k, v, q_lens=ql, k_lens=kl,
        q_scale=0.7, softmax_scale=0.2, causal=True, dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('case', ['dense', 'ragged', 'cross', 'causal', 'unequal_causal', 'window', 'empty'])
def test_musa_flash_forward_backward_against_cpu(monkeypatch, case):
    if not hasattr(torch, 'musa') or not torch.musa.is_available():
        pytest.skip('MUSA device required')
    from worldscape_policy.wam.wan22 import musa_attention as module
    if module._flash_varlen() is None:
        pytest.skip('MUSA FlashAttention-2 extension required')
    monkeypatch.setenv('WSP_MUSA_ATTENTION_BACKEND', 'flash_attn')
    torch.manual_seed(11)
    lq, lk = (9, 13) if case in {'cross', 'unequal_causal'} else (13, 13)
    q = torch.randn(2, lq, 2, 128).bfloat16().float().requires_grad_()
    k = torch.randn(2, lk, 2, 128).bfloat16().float().requires_grad_()
    v = torch.randn(2, lk, 2, 128).bfloat16().float().requires_grad_()
    kw = dict(softmax_scale=0.12, q_scale=0.8, causal=case in {'causal','unequal_causal'})
    if case in {'ragged','empty'}:
        kw.update(q_lens=torch.tensor([lq, 5]), k_lens=torch.tensor([lk, 0 if case=='empty' else 7]))
    if case == 'window':
        kw['window_size'] = (2, 1)
    expected = variable_length_sdpa(q, k, v, **kw, dtype=torch.float32)
    inputs = [x.detach().to('musa', torch.bfloat16).requires_grad_() for x in (q,k,v)]
    actual = module.musa_attention(*inputs, **kw)
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.025, rtol=0.025)
    grad = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, inputs, grad)
    refs = torch.autograd.grad(expected, (q,k,v), grad.float().cpu())
    for g, ref in zip(grads, refs):
        torch.testing.assert_close(g.float().cpu(), ref, atol=0.035, rtol=0.04)
