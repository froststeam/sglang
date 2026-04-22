import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell, is_musa

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if is_musa():
        if sm_version < 31:
            return False
    elif sm_version < 90:
        return False

    try:
        if not is_musa():
            import deep_gemm  # noqa: F401
        else:
            import mate.deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


def _get_deep_gemm_block_m() -> int:
    return envs.SGLANG_DEEP_GEMM_BLOCK_M.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_BLOCK_M = _get_deep_gemm_block_m()
