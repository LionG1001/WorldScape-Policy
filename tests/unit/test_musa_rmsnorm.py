import pytest
import torch

from worldscape_policy.wam.wan22.blocks import WanRMSNorm
from worldscape_policy.wam.wan22.musa_norm import musa_fused_rms_norm


def test_cpu_and_unsupported_inputs_preserve_original(monkeypatch):
    monkeypatch.setenv('WSP_MUSA_FUSED_RMSNORM', '1')
    layer = WanRMSNorm(16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    expected = layer._norm(x.float()).type_as(x) * layer.weight
    torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)
    assert musa_fused_rms_norm(x, layer.eps) is None


@pytest.mark.parametrize('kind', ['random', 'zero', 'small', 'strided'])
def test_musa_output_input_and_weight_gradients(monkeypatch, kind):
    if not hasattr(torch, 'musa') or not torch.musa.is_available():
        pytest.skip('MUSA device required')
    from worldscape_policy.wam.wan22 import musa_norm
    if not musa_norm._available():
        pytest.skip('Native MUSA RMSNorm forward/backward unavailable')
    torch.manual_seed(19)
    layer = WanRMSNorm(128).to('musa', dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.uniform_(.5, 1.5)
    source = torch.randn(2, 7, 128, device='musa', dtype=torch.bfloat16)
    if kind == 'zero':
        source.zero_()
    elif kind == 'small':
        source *= 1e-4
    elif kind == 'strided':
        source = torch.randn(2, 7, 256, device='musa', dtype=torch.bfloat16)[..., ::2]
        assert musa_fused_rms_norm(source, layer.eps) is None
    incoming = torch.randn_like(source)
    observed = []
    for enabled in ('0', '1', '0'):
        monkeypatch.setenv('WSP_MUSA_FUSED_RMSNORM', enabled)
        # Preserve strides to exercise the fallback as well as the fast path.
        x = source.detach().requires_grad_()
        result = layer(x)
        grads = torch.autograd.grad(result, (x, layer.weight), incoming)
        values = [t.detach().float().cpu() for t in (result, *grads)]
        if observed:
            for value, reference in zip(values, observed[0]):
                assert torch.isfinite(value).all()
                error = (value - reference).norm() / reference.norm().clamp_min(1e-12)
                assert error <= .01, float(error)
        observed.append(values)


def test_musa_missing_operator_falls_back(monkeypatch):
    if not hasattr(torch, 'musa') or not torch.musa.is_available():
        pytest.skip('MUSA device required')
    from worldscape_policy.wam.wan22 import musa_norm
    monkeypatch.setenv('WSP_MUSA_FUSED_RMSNORM', '1')
    monkeypatch.setattr(musa_norm, '_available', lambda: False)
    layer = WanRMSNorm(128).to('musa', dtype=torch.bfloat16)
    x = torch.randn(2, 7, 128, device='musa', dtype=torch.bfloat16)
    torch.testing.assert_close(layer(x), layer._norm(x.float()).type_as(x) * layer.weight, atol=0, rtol=0)


def test_musa_operator_errors_are_not_silenced(monkeypatch):
    if not hasattr(torch, 'musa') or not torch.musa.is_available():
        pytest.skip('MUSA device required')
    from worldscape_policy.wam.wan22 import musa_norm
    monkeypatch.setattr(musa_norm, '_available', lambda: True)
    def fail(*args):
        raise RuntimeError('operator failure')
    monkeypatch.setattr(torch.ops.aten, '_fused_rms_norm', fail)
    x = torch.ones(2, 128, device='musa', dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='operator failure'):
        musa_fused_rms_norm(x, 1e-6)
