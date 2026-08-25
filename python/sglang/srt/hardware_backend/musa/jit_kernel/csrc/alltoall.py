from __future__ import annotations

import functools

import torch

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit


@functools.lru_cache(maxsize=1)
def _custom_a2a_module():
    return load_musa_jit(
        "sglang_musa_custom_all_to_all",
        ("distributed/custom_all_to_all.mu",),
        extra_musa_cflags=(
            "-Wno-error=address-of-temporary",
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-D__MUSA_ARCH_LIST__=310",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
        ),
    )


def ensure_compiled() -> None:
    _custom_a2a_module()


def meta_size() -> int:
    return int(_custom_a2a_module().sgl_musa_custom_a2a_meta_size())


def launch(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    input_: torch.Tensor,
    output: torch.Tensor,
    self_signal_ptr: int,
    slot_stride_bytes: int,
    slot: int,
    slots: int,
    rank: int,
    world_size: int,
) -> None:
    _custom_a2a_module().sgl_musa_custom_a2a_launch(
        rank_data,
        signal_ptrs_cpu,
        input_,
        output,
        int(self_signal_ptr),
        int(slot_stride_bytes),
        int(slot),
        int(slots),
        int(rank),
        int(world_size),
    )

def launch_ulysses(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    input_: torch.Tensor,
    output: torch.Tensor,
    self_signal_ptr: int,
    slot_stride_bytes: int,
    slot: int,
    slots: int,
    local_sequence: int,
    rank: int,
    world_size: int,
    input_layout: bool,
) -> None:
    _custom_a2a_module().sgl_musa_custom_ulysses_launch(
        rank_data,
        signal_ptrs_cpu,
        input_,
        output,
        int(self_signal_ptr),
        int(slot_stride_bytes),
        int(slot),
        int(slots),
        int(local_sequence),
        int(rank),
        int(world_size),
        int(input_layout),
    )


def launch_qkv_ulysses(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_output: torch.Tensor,
    key_output: torch.Tensor,
    value_output: torch.Tensor,
    self_signal_ptr: int,
    slot_stride_bytes: int,
    query_slot: int,
    key_slot: int,
    value_slot: int,
    slots: int,
    local_sequence: int,
    rank: int,
    world_size: int,
) -> None:
    _custom_a2a_module().sgl_musa_custom_qkv_ulysses_launch(
        rank_data,
        signal_ptrs_cpu,
        query,
        key,
        value,
        query_output,
        key_output,
        value_output,
        int(self_signal_ptr),
        int(slot_stride_bytes),
        int(query_slot),
        int(key_slot),
        int(value_slot),
        int(slots),
        int(local_sequence),
        int(rank),
        int(world_size),
    )


def launch_ulysses_prefix_output(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    prefix: torch.Tensor,
    sharded: torch.Tensor,
    output: torch.Tensor,
    self_signal_ptr: int,
    slot_stride_bytes: int,
    slot: int,
    slots: int,
    prefix_sequence: int,
    local_sequence: int,
    rank: int,
    world_size: int,
) -> None:
    _custom_a2a_module().sgl_musa_custom_ulysses_prefix_output_launch(
        rank_data,
        signal_ptrs_cpu,
        prefix,
        sharded,
        output,
        int(self_signal_ptr),
        int(slot_stride_bytes),
        int(slot),
        int(slots),
        int(prefix_sequence),
        int(local_sequence),
        int(rank),
        int(world_size),
    )
