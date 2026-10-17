"""Batch grouping preserves sample order and independent temporal histories."""
import torch
import pytest
from types import SimpleNamespace
from worldscape_policy.wam.wan22.vae import WanVideoVAE


class TemporalEncoder(WanVideoVAE):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def single_encode(self, video):
        # A causal transform makes accidental mixing of B and T observable.
        return video.cumsum(dim=2) * 0.25


def test_grouping_preserves_order_tail_and_input_gradient():
    encoder = TemporalEncoder()
    original = torch.arange(5*2*7*2*3, dtype=torch.float32).reshape(5,2,7,2,3)
    reference_input = original.clone().requires_grad_()
    reference = encoder.encode(reference_input)
    reference.square().sum().backward()
    for size in (1,2,3,5,8):
        candidate_input = original.clone().requires_grad_()
        result = encoder._encode_batched(candidate_input, size)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        result.square().sum().backward()
        torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=0, atol=0)


def test_cpu_path_is_not_changed_by_musa_setting(monkeypatch):
    encoder = TemporalEncoder()
    monkeypatch.setenv('WSP_MUSA_VAE_BATCH_SIZE', '4')
    def unexpected(*args):
        raise AssertionError('MUSA-only path entered on CPU')
    monkeypatch.setattr(encoder, '_encode_batched', unexpected)
    x = torch.randn(3,2,7,2,3)
    torch.testing.assert_close(encoder.encode(x), x.cumsum(dim=2)*.25)


class MusaDispatchTensor(torch.Tensor):
    """CPU storage with only the Python dispatch tag mocked; not a GPU test."""
    @property
    def device(self):
        return SimpleNamespace(type="musa")


@pytest.mark.parametrize("group,expected", [(0, [1]*5), (1, [1]*5), (3, [3, 2])])
def test_public_musa_dispatch_and_opt_out(monkeypatch, group, expected):
    encoder = TemporalEncoder()
    calls = []
    original = encoder.single_encode
    def record(x):
        calls.append(x.shape[0])
        return original(x)
    monkeypatch.setattr(encoder, "single_encode", record)
    monkeypatch.setenv("WSP_MUSA_VAE_BATCH_SIZE", str(group))
    x = torch.randn(5, 2, 7, 2, 3)
    result = encoder.encode(x.as_subclass(MusaDispatchTensor)).as_subclass(torch.Tensor)
    assert calls == expected
    torch.testing.assert_close(result, x.cumsum(dim=2)*.25)


def test_tiled_path_does_not_enter_grouping(monkeypatch):
    encoder = TemporalEncoder()
    encoder.upsampling_factor = 1
    monkeypatch.setenv("WSP_MUSA_VAE_BATCH_SIZE", "16")
    def unexpected(*args):
        raise AssertionError("Tiled input entered grouped path")
    monkeypatch.setattr(encoder, "_encode_batched", unexpected)
    monkeypatch.setattr(encoder, "tiled_encode", lambda x, *args: x)
    x = torch.randn(2, 2, 7, 2, 3)
    result = encoder.encode(x.as_subclass(MusaDispatchTensor), tiled=True).as_subclass(torch.Tensor)
    torch.testing.assert_close(result, x)
