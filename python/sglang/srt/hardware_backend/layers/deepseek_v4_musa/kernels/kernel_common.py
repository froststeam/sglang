import os
from functools import lru_cache

import torch

# Fixed-size bitset buffer in global memory for _tilelang_topk_transform_512_select_kernel.
# TileLang's type parser only resolves literal integers in T.Tensor shape annotations,
# not local/closure variables. 8200 * 32 = 262400 indices; P8 production max is 262208
# (and may grow), so we pad to 8200 to cover 262400 indices and leave headroom.
# This avoids the "no available layout found" error that occurs with T.alloc_shared
# for very large bitsets.
SELECT_TOPK_BITSET_WORDS = 8200

_TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]

_TILELANG_MUSA_DSA_FULL_DEVICE_COMPILE_FLAGS = [
    *_TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS,
    "-mllvm",
    "-mtgpu-combine-instr-with-burst=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

_TILELANG_MUSA_OPT1_DEVICE_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-opt-level=1",
]

_TILELANG_MUSA_LS_DEVICE_COMPILE_FLAGS = [
    *_TILELANG_MUSA_OPT1_DEVICE_COMPILE_FLAGS,
    "-mllvm",
    "-mtgpu-load-store-opt=1",
    "-mllvm",
    "-mtgpu-fold-global-ldst=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "-mtgpu-store-cluster-mutation=1",
    "-mllvm",
    "-mtgpu-memory-sched-mutation=1",
]

def _tilelang_musa_compile_profile_flags(default_profile: str | None = None) -> list[str] | None:
    profile = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_TILELANG_COMPILE_PROFILE", "").strip().lower()
    if profile == "" and default_profile is not None:
        profile = default_profile.strip().lower()
    if profile in {"", "default", "none", "0"}:
        return None
    if profile == "opt1":
        return _TILELANG_MUSA_OPT1_DEVICE_COMPILE_FLAGS
    if profile == "ls":
        return _TILELANG_MUSA_LS_DEVICE_COMPILE_FLAGS
    if profile == "dsa":
        return _TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS
    if profile == "dsa_full":
        return _TILELANG_MUSA_DSA_FULL_DEVICE_COMPILE_FLAGS
    raise ValueError(
        "Unsupported SGLANG_DEEPSEEK_V4_MUSA_TILELANG_COMPILE_PROFILE="
        f"{profile!r}; expected one of default,opt1,ls,dsa,dsa_full"
    )

def _tilelang_musa_pass_configs(tilelang, *, compile_profile: str | None = None):
    compile_flags = _tilelang_musa_compile_profile_flags(compile_profile)
    pass_configs = {}
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_TILELANG_PASS_CONFIG") == "1":
        pass_configs.update(
            {
                tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
                tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
            }
        )
    elif compile_flags is None:
        return None
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_TILELANG_AGGRESSIVE_PASS_CONFIG") == "1":
        pass_configs.update(
            {
                tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
                tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
                tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
                tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
            }
        )
    # Keep index promotion enabled by default. Large-model page/cache tensors can
    # exceed INT32 element-address ranges; only opt into this pass after a kernel
    # has explicit INT64 address casts or generated-code validation.
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_DISABLE_INDEX_PROMOTION") == "1":
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    if compile_flags is not None:
        pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = compile_flags
    return pass_configs

def _tilelang_musa_burst_reduce_pass_configs(tilelang, *, compile_profile: str | None = None):
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    }
    compile_flags = _tilelang_musa_compile_profile_flags(compile_profile)
    if compile_flags is not None:
        pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = compile_flags
    return pass_configs

def _tilelang_musa_reduce_profile_pass_configs(
    tilelang,
    *,
    compile_profile: str | None = None,
    reduce_profile: str = "burst",
):
    profile = (reduce_profile or "burst").strip().lower()
    if profile in {"burst", "reduce_burst", "default", "1", "true"}:
        return _tilelang_musa_burst_reduce_pass_configs(
            tilelang,
            compile_profile=compile_profile,
        )
    if profile in {"noburst", "no_burst", "plain", "0", "false"}:
        pass_configs = {tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True}
        compile_flags = _tilelang_musa_compile_profile_flags(compile_profile)
        if compile_flags is not None:
            pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = compile_flags
        return pass_configs
    raise ValueError(
        "Unsupported TileLang MUSA reduce_profile="
        f"{reduce_profile!r}; expected burst or noburst"
    )

def _tilelang_musa_aggressive_pass_configs(
    tilelang,
    *,
    disable_index_promotion: bool = True,
    compile_profile: str | None = None,
):
    pass_configs = _tilelang_musa_burst_reduce_pass_configs(
        tilelang,
        compile_profile=compile_profile,
    )
    pass_configs.update(
        {
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
            tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
            tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
        }
    )
    if disable_index_promotion:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_TILELANG_DISABLE_HOST_ASSERTS") == "1":
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_HOST_ASSERTS] = True
    compile_flags = _tilelang_musa_compile_profile_flags(compile_profile)
    if compile_flags is not None:
        pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = compile_flags
    return pass_configs

def _tilelang_musa_dsa_pass_configs(
    tilelang,
    *,
    full: bool = False,
    disable_index_promotion: bool = True,
):
    pass_configs = _tilelang_musa_aggressive_pass_configs(
        tilelang,
        disable_index_promotion=disable_index_promotion,
    )
    pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = (
        _TILELANG_MUSA_DSA_FULL_DEVICE_COMPILE_FLAGS
        if full
        else _TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS
    )
    return pass_configs

def _tilelang_jit(tilelang, name: str, pass_configs=None):
    if pass_configs is None:
        pass_configs = _tilelang_musa_pass_configs(tilelang)
    try:
        if pass_configs is None:
            return tilelang.jit(name=name)
        return tilelang.jit(name=name, pass_configs=pass_configs)
    except TypeError as exc:
        if "name" not in str(exc) and "pass_configs" not in str(exc):
            raise
        if pass_configs is None:
            return tilelang.jit
        return tilelang.jit(pass_configs=pass_configs)
