from types import SimpleNamespace
import torch
from worldscape_policy.training.musa_zero import _aligned_flatten


def test_padding_preserves_parameters_and_per_rank_alignment():
    optimizer=SimpleNamespace(nccl_start_alignment_factor=2, allgather_bucket_size=500000000)
    values=[torch.arange(13,dtype=torch.bfloat16),torch.arange(6,dtype=torch.bfloat16)]
    def original(self,tensors,alignment,use_cpu_data=False):
        flat=torch.cat(tensors)
        return torch.nn.functional.pad(flat,(0,(-flat.numel())%alignment))
    flat=_aligned_flatten(original,optimizer,values,2*16,False,128)
    assert optimizer.nccl_start_alignment_factor==64
    assert flat.numel()==16*64
    torch.testing.assert_close(flat[:19],torch.cat(values))
    assert torch.count_nonzero(flat[19:])==0
    # Subsequent parameter groups already receive the updated alignment.
    second=_aligned_flatten(original,optimizer,values,64*16,False,128)
    assert second.numel()==flat.numel()


def test_checkpoint_alignment_mismatch_is_rejected_before_engine_load(tmp_path):
    import json
    import pytest
    from worldscape_policy.training.runtime import NativeTrainingRuntime, NativeDistributedConfig
    runtime = NativeTrainingRuntime(NativeDistributedConfig(backend="single", device="cpu"))
    runtime.engine = SimpleNamespace(optimizer=SimpleNamespace(_wsp_alignment_bytes=128))
    with pytest.raises(ValueError, match="alignment differs"):
        runtime.load_deepspeed_checkpoint(str(tmp_path), tag="step")
    runtime.engine.optimizer._wsp_alignment_bytes = 0
    (tmp_path / "step").mkdir()
    (tmp_path / "step" / "wsp-zero-alignment.json").write_text(json.dumps({"alignment_bytes":128}))
    with pytest.raises(ValueError, match="alignment differs"):
        runtime.load_deepspeed_checkpoint(str(tmp_path), tag="step")


def test_cpu_staging_checks_real_storage_not_placeholder_dtype():
    optimizer = SimpleNamespace(nccl_start_alignment_factor=2)
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    parameter.cpu_data = torch.arange(19, dtype=torch.bfloat16)
    def original(self, tensors, alignment, use_cpu_data=False):
        assert use_cpu_data
        flat = torch.cat([p.cpu_data for p in tensors])
        return torch.nn.functional.pad(flat, (0, (-flat.numel()) % alignment))
    flat = _aligned_flatten(original, optimizer, [parameter], 32, True, 128)
    torch.testing.assert_close(flat[:19], parameter.cpu_data)
    assert flat.numel() == 1024
