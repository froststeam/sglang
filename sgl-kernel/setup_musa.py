# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch_musa
import torch_musa.utils.musa_extension as musa_ext
import torch_musa.utils.simple_porting as musa_sp
from setuptools import find_packages, setup

root = Path(__file__).parent.resolve()
third_party = Path("third_party")
arch = platform.machine().lower()


# Override musa_ex and musa_sp to recognize .cu/.cuh files
def _custom_is_musa_file(path: str) -> bool:
    return os.path.splitext(path)[1] in [".cu", ".cuh", ".mu", ".muh"]


if hasattr(musa_ext, "_is_musa_file"):
    musa_ext._is_musa_file = _custom_is_musa_file

if hasattr(musa_sp, "EXT_REPLACED_MAPPING"):
    musa_sp.EXT_REPLACED_MAPPING = {"cuh": "cuh", "cu": "cu"}


class _RepoInfo:
    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_FLASHINFER_REPO = _RepoInfo(
    name="flashinfer",
    git_repository="git@sh-code.mthreads.com:mcc-ai-serving/flashinfer.git",
    git_tag="bc29697ba20b7e6bdb728ded98f04788e16ee021",
    git_shallow=False,
)

_MUTLASS_REPO = _RepoInfo(
    name="mutlass",
    git_repository="git@sh-code.mthreads.com:sw/mutlass.git",
    git_tag="5edb98674a92af0d09d1ebd9bbdaab907b95a91c",
    git_shallow=False,
)

_HADAMARD_REPO = _RepoInfo(
    name="fast-hadamard-transform",
    git_repository="https://github.com/sgl-project/fast-hadamard-transform.git",
    git_tag="48f3c13764dc2ec662ade842a4696a90a137f1bc",
    git_shallow=False,
)


class _CustomBuildExt(musa_ext.BuildExtension):
    # define a const to set common mapping rules
    _MAPPING_RULE = {
        # ATen
        "#include <ATen/cuda/Atomic.cuh>": '#include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/Atomic.muh"',
        "#include <ATen/cuda/CUDAContext.h>": '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',
        "#include <ATen/cuda/CUDADataType.h>": '#include "torch_musa/csrc/aten/musa/MUSADtype.muh"',
        "#include <ATen/cuda/CUDAGeneratorImpl.h>": '#include "torch_musa/csrc/aten/musa/CUDAGeneratorImpl.h"',
        "#include <ATen/cuda/detail/UnpackRaw.cuh>": '#include "torch_musa/csrc/aten/musa/UnpackRaw.muh"',
        "#include <ATen/cuda/Exceptions.h>": '#include "torch_musa/csrc/aten/musa/Exceptions.h"',
        "at::cuda": "at::musa",
        "torch::kCUDA": "torch::kMUSA",
        # C10
        "#include <c10/cuda/CUDAException.h>": '#include "torch_musa/csrc/core/MUSAException.h"',
        "#include <c10/cuda/CUDAGuard.h>": '#include "torch_musa/csrc/core/MUSAGuard.h"',
        "#include <c10/cuda/CUDAStream.h>": '#include "torch_musa/csrc/core/MUSAStream.h"',
        "c10::cuda": "c10::musa",
        "C10_CUDA_KERNEL_LAUNCH_CHECK": "C10_MUSA_KERNEL_LAUNCH_CHECK",
        "C10_CUDA_CHECK": "C10_MUSA_CHECK",
        # CUDA
        "curandStatePhilox4_32_10_t": "murandStatePhilox4_32_10_t",
        "curand_init": "murand_init",
        "curand_uniform": "murand_uniform",
        "curand_uniform4": "murand_uniform4",
        "cudaLaunchAttribute": "musaLaunchAttribute",
        "cudaLaunchAttributeProgrammaticStreamSerialization": "musaLaunchAttributeIgnore",  # XXX (MUSA): not supported
        "cudaLaunchConfig_t": "musaLaunchConfig_t",
        "cuPointerGetAttribute": "muPointerGetAttribute",
        "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR": "MU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
        "CUdeviceptr": "MUdeviceptr",
        # cuBLAS
        "CUBLASLT_MATMUL_DESC_A_SCALE_POINTER": "MUBLASLT_MATMUL_DESC_A_SCALE_POINTER",
        "CUBLASLT_MATMUL_DESC_B_SCALE_POINTER": "MUBLASLT_MATMUL_DESC_B_SCALE_POINTER",
        "CUBLASLT_MATMUL_DESC_FAST_ACCUM": "MUBLASLT_MATMUL_DESC_FAST_ACCUM",
        "CUBLASLT_MATMUL_DESC_TRANSA": "MUBLASLT_MATMUL_DESC_TRANSA",
        "CUBLASLT_MATMUL_DESC_TRANSB": "MUBLASLT_MATMUL_DESC_TRANSB",
        "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES": "MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",
        "CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT": "MUBLASLT_MATRIX_LAYOUT_BATCH_COUNT",
        "CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET": "MUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",
        "CUBLAS_COMPUTE_32F": "MUBLAS_COMPUTE_32F",
        "CUBLAS_OP_N": "MUBLAS_OP_N",
        "CUBLAS_OP_T": "MUBLAS_OP_T",
        "CUBLAS_STATUS_NOT_SUPPORTED": "MUBLAS_STATUS_NOT_IMPLEMENTED",
        "CUBLAS_STATUS_SUCCESS": "MUBLAS_STATUS_SUCCESS",
        "cublasComputeType_t": "mublasComputeType_t",
        "cublasGetStatusString": "mublasGetStatusString",
        "cublasLtHandle_t": "mublasLtHandle_t",
        "cublasLtMatmul": "mublasLtMatmul",
        "cublasLtMatmulAlgoGetHeuristic": "mublasLtMatmulAlgoGetHeuristic",
        "cublasLtMatmulDescAttributes_t": "mublasLtMatmulDescAttributes_t",
        "cublasLtMatmulDescCreate": "mublasLtMatmulDescCreate",
        "cublasLtMatmulDescDestroy": "mublasLtMatmulDescDestroy",
        "cublasLtMatmulDescOpaque_t": "mublasLtMatmulDescOpaque_t",
        "cublasLtMatmulDescSetAttribute": "mublasLtMatmulDescSetAttribute",
        "cublasLtMatmulDesc_t": "mublasLtMatmulDesc_t",
        "cublasLtMatmulHeuristicResult_t": "mublasLtMatmulHeuristicResult_t",
        "cublasLtMatmulPreferenceAttributes_t": "mublasLtMatmulPreferenceAttributes_t",
        "cublasLtMatmulPreferenceCreate": "mublasLtMatmulPreferenceCreate",
        "cublasLtMatmulPreferenceDestroy": "mublasLtMatmulPreferenceDestroy",
        "cublasLtMatmulPreferenceOpaque_t": "mublasLtMatmulPreferenceOpaque_t",
        "cublasLtMatmulPreferenceSetAttribute": "mublasLtMatmulPreferenceSetAttribute",
        "cublasLtMatmulPreference_t": "mublasLtMatmulPreference_t",
        "cublasLtMatrixLayoutAttribute_t": "mublasLtMatrixLayoutAttribute_t",
        "cublasLtMatrixLayoutCreate": "mublasLtMatrixLayoutCreate",
        "cublasLtMatrixLayoutDestroy": "mublasLtMatrixLayoutDestroy",
        "cublasLtMatrixLayoutOpaque_t": "mublasLtMatrixLayoutOpaque_t",
        "cublasLtMatrixLayoutSetAttribute": "mublasLtMatrixLayoutSetAttribute",
        "cublasLtMatrixLayout_t": "mublasLtMatrixLayout_t",
        "cublasStatus_t": "mublasStatus_t",
        # Data types
        "__NV_E4M3": "__MT_E4M3",
        "__NV_E5M2": "__MT_E5M2",
        "__NV_SATFINITE": "__MT_SATFINITE",
        "__nv_bfloat16": "__mt_bfloat16",
        "__nv_bfloat162": "__mt_bfloat162",
        "__nv_cvt_float2_to_fp8x2": "__musa_cvt_float2_to_fp8x2",
        "__nv_fp8_e4m3": "__mt_fp8_e4m3",
        "__nv_fp8_e5m2": "__mt_fp8_e5m2",
        "__nv_fp8x2_e4m3": "__mt_fp8x2_e4m3",
        "__nv_fp8x2_e5m2": "__mt_fp8x2_e5m2",
        "__nv_fp8x2_storage_t": "__mt_fp8x2_storage_t",
        "__nv_fp8x4_e4m3": "__mt_fp8x4_e4m3",
        "__nv_fp8x4_e5m2": "__mt_fp8x4_e5m2",
        "__nv_fp8x4_storage_t": "__mt_fp8x4_storage_t",
        "nv_bfloat16": "__mt_bfloat16",
        "nv_bfloat162": "__mt_bfloat162",
        "__nv_half": "__half",
        "nv_half": "__half",
        # Others
        "#include <cuda_fp8.h>": "#include <musa_fp8.h>",
        ".FlagHeads<VEC_SIZE>": ".template FlagHeads<VEC_SIZE>",
        ".InclusiveSum<VEC_SIZE>": ".template InclusiveSum<VEC_SIZE>",
        ".Reduce<VEC_SIZE>": ".template Reduce<VEC_SIZE>",
        ".Sum<VEC_SIZE>": ".template Sum<VEC_SIZE>",
        "CUDA_R_8F_E4M3": "MUSA_R_8F_E4M3",
        "CUDA_R_8F_E5M2": "MUSA_R_8F_E5M2",
        # THC
        "#include <THC/THCAtomics.cuh>": "#include <THC/THCAtomics.muh>",
        # Cutlass
        '#include "cutlass/array.h"': "#include <mutlass/array.h>",
        "#include <cutlass/array.h>": "#include <mutlass/array.h>",
        "#include <cutlass/cutlass.h>": "#include <mutlass/mutlass.h>",
        "#include <cutlass/numeric_types.h>": "#include <mutlass/numeric_types.h>",
        "cutlass::AlignedArray": "mutlass::AlignedArray",
        "cutlass::bfloat16_t": "mutlass::bfloat16_t",
        "cutlass::half_t": "mutlass::half_t",
    }

    def build_extensions(self):
        self.compiler.src_extensions += [".cu", ".cuh"]

        super().build_extensions()

    def run(self):
        def clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
            repo_path.parent.mkdir(exist_ok=True)
            if not repo_path.exists():
                clone_cmd = ["git", "clone"]
                if git_shallow:
                    clone_cmd += ["--depth", "1"]
                clone_cmd += [repo_url, str(repo_path)]
                subprocess.check_call(clone_cmd)
                subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
            else:
                subprocess.check_call(["git", "fetch", "--all"], cwd=repo_path)
                subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)

        if os.environ.get("SKIP_THIRD_PARTY", "0") == "1":
            print("Skipping third-party repositories cloning as requested.")
        else:
            clone_and_checkout(
                _MUTLASS_REPO.source_dir,
                _MUTLASS_REPO.git_repository,
                _MUTLASS_REPO.git_tag,
                _MUTLASS_REPO.git_shallow,
            )

            clone_and_checkout(
                _FLASHINFER_REPO.source_dir,
                _FLASHINFER_REPO.git_repository,
                _FLASHINFER_REPO.git_tag,
                _FLASHINFER_REPO.git_shallow,
            )

            clone_and_checkout(
                _HADAMARD_REPO.source_dir,
                _HADAMARD_REPO.git_repository,
                _HADAMARD_REPO.git_tag,
                _HADAMARD_REPO.git_shallow,
            )

        fast_math = """
static __device__ __forceinline__ float fast_rsqrtf(float a) {
  float x = 0.5f * a;
  float y = __frsqrt_rn(a);
  y = y * (1.5f - x * y * y);
  return y;
}

static __device__ __forceinline__ float fast_rcp(float a) {
  return __frcp_rn(a);
}
"""

        musa_sp.SimplePorting(
            cuda_dir_path=_FLASHINFER_REPO.source_dir / "include",
            mapping_rule={
                **self._MAPPING_RULE,
                # include/flashinfer/vec_dtypes.cuh
                "(__CUDA_ARCH__ < 800)": "(__MUSA_ARCH__ < 220)",  # Guard to skip __hmul/__hmul2 redefinitions
                "(__CUDA_ARCH__ >= 900)": "(__MUSA_ARCH__ >= 310)",  # Define FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
                "::cast<vec_size>": "::template cast<vec_size>",  # mcc compiler requires 'template' keyword
                # include/flashinfer/norm.cuh
                '#include "math.cuh"': fast_math,
                "math::shfl_xor_sync(sum_sq, offset);": "__shfl_xor_sync(0xffffffff, sum_sq, offset);",
                "math::rsqrt(smem[0] / float(d) + eps);": "fast_rsqrtf(smem[0] / float(d) + eps);",
                # include/flashinfer/sampling.cuh
                "math::ptx_rcp(max(sum_low, 1e-8));": "fast_rcp(max(sum_low, 1e-8));",
                "math::ptx_rcp(denom);": "fast_rcp(denom);",
                "#include <cuda/functional>": "#include <musa/functional>",
                "#include <cuda/std/functional>": "#include <musa/std/functional>",
                "#include <cuda/std/limits>": "#include <musa/std/limits>",
                "cuda::std::numeric_limits": "musa::std::numeric_limits",
                "compute_capacity.first >= 8": "compute_capacity.first >= 3",
            },
        ).run()

        musa_sp.SimplePorting(
            cuda_dir_path=_FLASHINFER_REPO.source_dir / "csrc",
            mapping_rule={
                **self._MAPPING_RULE,
                # csrc/pytorch_extension_utils.h
                ".is_cuda()": ".is_privateuseone()",
                # csrc/sampling.cu
                "->philox_cuda_state": "->philox_musa_state",
            },
        ).run()

        musa_sp.SimplePorting(
            cuda_dir_path=_HADAMARD_REPO.source_dir / "csrc",
            mapping_rule={
                **self._MAPPING_RULE,
                # csrc/pytorch_extension_utils.h
                ".is_cuda()": ".is_privateuseone()",
            },
        ).run()

        musa_sp.SimplePorting(
            cuda_dir_path="csrc",
            mapping_rule={
                **self._MAPPING_RULE,
                # sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu
                "SCHEDULER::execute": "SCHEDULER::template execute",
                # sgl-kernel/csrc/elementwise/topk.cu
                ".is_cuda()": ".is_privateuseone()",
            },
        ).run()

        super().run()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
    root / _FLASHINFER_REPO.source_dir / "include_musa",
    root / _FLASHINFER_REPO.source_dir / "csrc_musa",
    root / _HADAMARD_REPO.source_dir / "csrc_musa",
    root / _MUTLASS_REPO.source_dir / "include",
]

sources = [
    "csrc_musa/allreduce/custom_all_reduce.cu",
    "csrc_musa/attention/merge_attn_states.cu",
    "csrc_musa/common_extension_musa.cc",
    "csrc_musa/elementwise/activation.cu",
    "csrc_musa/elementwise/concat_mla.mu",
    "csrc_musa/elementwise/fused_add_rms_norm_kernel.mu",
    "csrc_musa/elementwise/topk.cu",
    "csrc_musa/elementwise/pos_enc.cu",
    "csrc_musa/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc_musa/moe/moe_align_kernel.cu",
    "csrc_musa/moe/moe_fused_gate_musa.cu",
    "csrc_musa/moe/kimi_k2_moe_fused_gate.cu",
    "csrc_musa/moe/moe_sum.cu",
    "csrc_musa/moe/moe_sum_reduce.cu",
    "csrc_musa/moe/moe_topk_softmax_kernels.cu",
    "csrc_musa/moe/moe_topk_sigmoid_kernels.cu",
    "csrc_musa/quantization/gguf/gguf_kernel.cu",
    "csrc_musa/speculative/eagle_utils.cu",
    "csrc_musa/speculative/ngram_utils.cu",
    "csrc_musa/speculative/packbit.cu",
    "csrc_musa/speculative/speculative_sampling.cu",
    "csrc_musa/memory/store.cu",
    "csrc_musa/kvcacheio/transfer.cu",
    "csrc_musa/gemm/awq_kernel.cu",
    "csrc_musa/gemm/bmm_fp8.cu",
    "csrc_musa/gemm/dsv3_fused_a_gemm.cu",
    "csrc_musa/gemm/dsv3_router_gemm_bf16_out.cu",
    "csrc_musa/gemm/dsv3_router_gemm_entry.cu",
    "csrc_musa/gemm/dsv3_router_gemm_float_out.cu",
    "csrc_musa/gemm/per_token_quant_fp8.cu",
    "csrc_musa/gemm/per_tensor_quant_fp8.cu",
    "csrc_musa/gemm/per_token_group_quant_8bit.cu",
    "csrc_musa/gemm/per_token_group_quant_8bit_v2.cu",
    str(_FLASHINFER_REPO.source_dir / "csrc_musa/norm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc_musa/renorm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc_musa/sampling.cu"),
    str(_HADAMARD_REPO.source_dir / "csrc_musa/fast_hadamard_transform_cuda.cu"),
    str(_HADAMARD_REPO.source_dir / "csrc_musa/fast_hadamard_transform.cpp"),
    # XXX (MUSA): The following files contain MUSA-specific implementations.
    "csrc_musa/musa/pos_encoding_contiguous.mu",
    "csrc_musa/musa/matmul_mudnn.cpp",
    "csrc_musa/musa/moe_gemv_swiglu.mu",
    "csrc_musa/musa/ternary.mu",
    "csrc_musa/musa/top_k_top_p_sampling.mu",
]

cxx_flags = ["force_mcc"]
libraries = ["c10", "torch", "torch_python"]
extra_link_args = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{arch}-linux-gnu",
    "-lmublasLt",
]

default_target = "mp_31"
mtgpu_target = os.environ.get("MTGPU_TARGET", default_target)

if torch.musa.is_available():
    try:
        prop = torch.musa.get_device_properties(0)
        mtgpu_target = f"mp_{prop.major}{prop.minor}"
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.musa not available. Using default target: {mtgpu_target}")

if mtgpu_target not in ["mp_22", "mp_31"]:
    print(
        f"Warning: Unsupported GPU architecture detected '{mtgpu_target}'. Expected 'mp_22' or 'mp_31'."
    )
    sys.exit(1)

# XXX (MUSA): These flags should NOT be published outside
mcc_private_use_flags = [
    "-mllvm",
    "-mtgpu-combine-instr-with-burst=0",
    "-mllvm",
    "-mtgpu-enchanced-minreg-schedule=1",
    "-mllvm",
    "-mtgpu-enable-preisel-sinking=0",
    "-mllvm",
    "-mtgpu-attr-alloc-first=1",
    "-mllvm",
    "-misched-recompute-slotindex",
]

mcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-fPIC",
    "-std=c++17",
    "-x",
    "musa",
    "-mtgpu",
    f"--cuda-gpu-arch={mtgpu_target}",
    "-fno-strict-aliasing",
    "-ffast-math",
    "-Od3",
    "-fmusa-flush-denormals-to-zero",
    "-DUSE_MUSA",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
]
mcc_flags += mcc_private_use_flags

if mtgpu_target == "mp_31":
    mcc_flags.extend(
        [
            "-DFLASHINFER_ENABLE_FP8",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
            "-DFLASHINFER_ENABLE_FP8_E5M2",
        ]
    )


ext_modules = [
    musa_ext.MUSAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "mcc": mcc_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
