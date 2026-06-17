import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_blackwell_supported,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

if _is_musa and not envs.SGLANG_DEEP_GEMM_BLOCK_M.is_set():
    # XXX (MUSA): Use a larger default block M for MUSA DeepGEMM MoE shapes,
    # while preserving explicit user overrides and the default on other backends.
    DEEPGEMM_BLOCK_M = 256
else:
    DEEPGEMM_BLOCK_M = envs.SGLANG_DEEP_GEMM_BLOCK_M.get()
DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
DEEPGEMM_SCALE_LAYOUT_COLUMN_MAJOR = not _is_musa
DEEPGEMM_SCALE_TMA_ALIGNED = not _is_musa
