"""Opt-in ZeRO-2 shard alignment, applied only during engine construction."""
from contextlib import contextmanager
import os
from unittest.mock import patch


def _aligned_flatten(original, optimizer, tensors, alignment, use_cpu_data, alignment_bytes):
    data = [t.cpu_data for t in tensors] if use_cpu_data else tensors
    if any(t.element_size() != 2 for t in data):
        raise ValueError('MUSA ZeRO alignment requires FP16/BF16 parameter groups')
    if not getattr(optimizer, '_wsp_alignment_bytes', 0):
        old_factor = optimizer.nccl_start_alignment_factor
        if old_factor != 2 or alignment % old_factor:
            raise RuntimeError('Unsupported DeepSpeed ZeRO alignment contract')
        world_size = alignment // old_factor
        factor = alignment_bytes // 2
        optimizer.nccl_start_alignment_factor = factor
        optimizer._wsp_alignment_bytes = alignment_bytes
        alignment = world_size * factor
    return original(optimizer, tensors, alignment, use_cpu_data=use_cpu_data)


@contextmanager
def musa_zero_alignment(device_type, stage):
    requested = int(os.environ.get('WSP_MUSA_ZERO_ALIGNMENT_BYTES', '0'))
    if not requested:
        yield
        return
    if requested not in {4, 8, 16, 32, 64, 128, 256}:
        raise ValueError('WSP_MUSA_ZERO_ALIGNMENT_BYTES must be 0 or a power of two from 4 to 256')
    if device_type != 'musa' or stage != 2:
        raise ValueError('MUSA shard alignment is supported only for MUSA ZeRO-2')
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    original = DeepSpeedZeroOptimizer.flatten_dense_tensors_aligned
    constructed = {}
    def flatten(optimizer, tensors, alignment, use_cpu_data=False):
        constructed[id(optimizer)] = optimizer
        return _aligned_flatten(original, optimizer, tensors, alignment, use_cpu_data, requested)
    # No installed-library edits or permanent global patch. The optimizer keeps
    # the alignment factor for subsequent all-gathers and checkpoint bookkeeping.
    with patch.object(DeepSpeedZeroOptimizer, 'flatten_dense_tensors_aligned', flatten):
        yield
    for optimizer in constructed.values():
        if optimizer.allgather_bucket_size % optimizer.nccl_start_alignment_factor:
            raise ValueError('allgather_bucket_size must be divisible by the requested alignment in elements')
