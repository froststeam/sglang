from __future__ import annotations

import os

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from mate.testing.utils import bench_kineto

from sglang.srt.hardware_backend.musa.jit_kernel.csrc.moe import (
    moe_sum_reduce as musa_jit_sum,
)


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    s0,
    s1,
    s2,
    output_ptr,
    os0,
    os1,
    token_num: tl.constexpr,
    topk_num: tl.constexpr,
    hidden_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask = (offs_m[:, None] < token_num) & (offs_d[None, :] < hidden_dim)
    base = input_ptr + offs_m[:, None] * s0 + offs_d[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_DIM), tl.float32)
    for k in tl.range(0, topk_num):
        v = tl.load(base + k * s1, mask=mask, other=0.0)
        acc += v.to(tl.float32)
    tl.store(
        output_ptr + offs_m[:, None] * os0 + offs_d[None, :],
        (acc * scale).to(input_ptr.dtype.element_ty),
        mask=mask,
    )


def triton_sum(x: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    m, topk, h = x.shape
    block_m = 1
    block_dim = 2048
    grid = (triton.cdiv(m, block_m), triton.cdiv(h, block_dim))
    _moe_sum_reduce_kernel[grid](
        x,
        *x.stride(),
        out,
        *out.stride(),
        token_num=m,
        topk_num=topk,
        hidden_dim=h,
        scale=scale,
        BLOCK_M=block_m,
        BLOCK_DIM=block_dim,
        num_warps=16,
    )


@torch.compile
def torch_compile_sum(x: torch.Tensor, out: torch.Tensor, scale: float):
    torch.sum(x * scale, dim=1, out=out)
    return out


def eager_ref(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sum(x.float() * scale, dim=1).to(x.dtype)


def bench_one(fn, kernel_names, multi: bool = False) -> float:
    ret = bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=8,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=multi,
    )
    return sum(ret) if isinstance(ret, tuple) else ret


def main() -> None:
    ms_env = os.environ.get("MS")
    ms = (
        [int(v) for v in ms_env.split(",")]
        if ms_env
        else [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
        ]
    )
    topk = int(os.environ.get("TOPK", "8"))
    hidden = int(os.environ.get("HIDDEN", "7168"))
    dtype = torch.bfloat16
    scale = 1.0
    print("device", torch.musa.current_device(), torch.musa.get_device_name())
    print(
        f"shape: M, topk={topk}, hidden={hidden}, "
        "dtype=bf16, flush_l2=True, num_tests=8"
    )
    print("bytes_lb = M * hidden * dtype_bytes * (topk + 1)")
    print(
        "M,jit_us,triton_us,torch_compile_us,"
        "jit_GBs,triton_GBs,torch_compile_GBs,"
        "maxdiff_jit,maxdiff_triton,maxdiff_tc"
    )
    for m in ms:
        torch.manual_seed(1234 + m)
        x = torch.randn((m, topk, hidden), device="musa", dtype=dtype).contiguous()
        out_jit = torch.empty((m, hidden), device="musa", dtype=dtype)
        out_tri = torch.empty_like(out_jit)
        out_tc = torch.empty_like(out_jit)

        musa_jit_sum(x, out_jit, scale)
        triton_sum(x, out_tri, scale)
        torch_compile_sum(x, out_tc, scale)
        torch.musa.synchronize()

        ref = eager_ref(x, scale)
        torch.musa.synchronize()
        diff_jit = (out_jit.float() - ref.float()).abs().max().item()
        diff_tri = (out_tri.float() - ref.float()).abs().max().item()
        diff_tc = (out_tc.float() - ref.float()).abs().max().item()
        if max(diff_jit, diff_tri, diff_tc) > 0.125:
            raise RuntimeError(
                f"bad diff m={m}: jit={diff_jit} triton={diff_tri} tc={diff_tc}"
            )

        if (
            (topk == 2 and m <= 8)
            or (
                topk == 4
                and (
                    m <= 8
                    or (m <= 32 and hidden <= 4096)
                    or (m >= 256 and hidden >= 7168)
                )
            )
            or (topk in (8, 9) and m <= 512)
        ):
            jit_name = (
                "moe_sum_reduce_scalar_vec2_topk_kernel"
                if ((topk == 4 and m >= 256) or (topk != 4 and m >= 64))
                else "moe_sum_reduce_scalar_topk_kernel"
            )
        elif m <= 96 and m > 32:
            jit_name = "moe_sum_reduce_small_token_vec8_kernel"
        elif m <= 32 and topk in (2, 4, 8, 9):
            jit_name = "moe_sum_reduce_small_token_vec8_topk_kernel"
        else:
            jit_name = "moe_sum_reduce_warp_token_vec8_kernel"

        jit_s = bench_one(lambda: musa_jit_sum(x, out_jit, scale), jit_name)
        triton_s = bench_one(
            lambda: triton_sum(x, out_tri, scale),
            "_moe_sum_reduce_kernel",
        )
        tc_s = bench_one(
            lambda: torch_compile_sum(x, out_tc, scale),
            ("triton_per_fused", "triton_poi_fused"),
            multi=True,
        )
        bytes_lb = m * hidden * x.element_size() * (topk + 1)

        def bw(seconds: float) -> float:
            return bytes_lb / seconds / 1e9 if seconds else 0.0

        print(
            f"{m},{jit_s * 1e6:.3f},{triton_s * 1e6:.3f},{tc_s * 1e6:.3f},"
            f"{bw(jit_s):.1f},{bw(triton_s):.1f},{bw(tc_s):.1f},"
            f"{diff_jit:.6g},{diff_tri:.6g},{diff_tc:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
