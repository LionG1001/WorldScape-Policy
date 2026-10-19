"""Optional native MUSA RMS normalization, keeping Wan's weight multiply separate."""
import functools
import logging

import torch

LOGGER = logging.getLogger(__name__)
_OBSERVED = set()


@functools.lru_cache(maxsize=1)
def _available():
    for name in ('aten::_fused_rms_norm', 'aten::_fused_rms_norm_backward'):
        try:
            if not torch._C._dispatch_has_kernel_for_dispatch_key(name, 'PrivateUse1'):
                return False
        except RuntimeError:
            # Older torch builds may not define these operators at all.
            return False
    return True


def musa_fused_rms_norm(x: torch.Tensor, eps: float):
    """Return None for unsupported inputs; runtime/operator failures propagate.

    The native operator accumulates BF16 normalization in FP32 and returns BF16.
    Do not pass the learned weight into it: Wan rounds the normalized tensor to
    the input dtype *before* its independent weight multiplication.
    """
    if (x.device.type != 'musa' or x.dtype != torch.bfloat16
            or x.ndim == 0 or x.numel() == 0 or not x.is_contiguous()):
        return None
    if not _available():
        if 'unavailable' not in _OBSERVED:
            LOGGER.warning('MUSA fused RMSNorm unavailable; using original Wan RMSNorm')
            _OBSERVED.add('unavailable')
        return None
    output, _ = torch.ops.aten._fused_rms_norm(x, [x.shape[-1]], None, eps)
    if 'native' not in _OBSERVED:
        LOGGER.info('MUSA RMSNorm path=native_fused weight_multiply=separate')
        _OBSERVED.add('native')
    return output
